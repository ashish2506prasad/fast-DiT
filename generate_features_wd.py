"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from pytorch_wavelets import DWTForward, DWTInverse
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import os
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
from glob import glob


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class CenterCropTransform:
    def __init__(self, image_size):
        self.image_size = image_size
    
    def __call__(self, pil_image):
        return center_crop_arr(pil_image, self.image_size)

# Create custom transform classes to avoid lambda functions
class CenterCropTransform:
    def __init__(self, image_size):
        self.image_size = image_size
    
    def __call__(self, pil_image):
        return center_crop_arr(pil_image, self.image_size)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    """
    Custom dataset to load images from a directory structure.
    Expects images to be in subdirectories: parent_dir/{class}/*/*.jpeg
    Splits data per class: 70% train, 10% val, 20% test
    
    Args:
        parent_dir (str): Path to the parent directory containing class subdirectories.
        split (str): One of 'train', 'val', or 'test'
        train_ratio (float): Fraction of data for training (default: 0.7)
        val_ratio (float): Fraction of data for validation (default: 0.1)
        image_size (int): Size to crop/resize images to (default: 256)
        
    Returns:
        torch.Tensor: Transformed image tensor.
    """
    def __init__(self, parent_dir, split='train', train_ratio=0.7, val_ratio=0.1, image_size=256):
        self.parent_dir = parent_dir
        
        # Get all class directories
        class_dirs = sorted([d for d in os.listdir(parent_dir) 
                           if os.path.isdir(os.path.join(parent_dir, d))])
        
        all_image_paths = []
        
        print(f"Processing {len(class_dirs)} classes for {split} split...")
        
        # Process each class separately
        for class_name in class_dirs:
            class_path = os.path.join(parent_dir, class_name)
            # Get all images for this class (both .jpeg and .JPEG)
            class_images = sorted(glob(f"{class_path}/*/*.jpeg") + 
                                glob(f"{class_path}/*/*.JPEG") +
                                glob(f"{class_path}/*/*.jpg") + 
                                glob(f"{class_path}/*/*.JPG"))
            
            if len(class_images) == 0:
                print(f"Warning: No images found for class {class_name}")
                continue
                
            # Split this class's data
            np.random.seed(42)  # For reproducibility
            indices = np.random.permutation(len(class_images))
            
            train_size = int(len(indices) * train_ratio)  # 70%
            val_size = int(len(indices) * val_ratio)      # 10%
            # test_size is the remaining 20%
            
            if split == 'train':
                selected_indices = indices[:train_size]
            elif split == 'val':
                selected_indices = indices[train_size:train_size + val_size]
            else:  # 'test'
                selected_indices = indices[train_size + val_size:]
            
            # Add selected images from this class
            class_selected_images = [class_images[i] for i in selected_indices]
            all_image_paths.extend(class_selected_images)
            
            print(f"Class {class_name}: {len(class_images)} total, {len(class_selected_images)} for {split}")
        
        self.image_paths = all_image_paths
        print(f"Total: {len(self.image_paths)} images in {parent_dir} for {split} split")
        
        # Setup transforms based on split
        if split == 'train':
            self.transform = transforms.Compose([
                CenterCropTransform(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True)
            ])
        else:  # 'val' or 'test'
            self.transform = transforms.Compose([
                CenterCropTransform(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True)
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            black_image = Image.new('RGB', (256, 256), (0, 0, 0))
            return self.transform(black_image)


def extract_dwt_features(latent, num_dwt_levels=1, device='cpu'):
    dwt = DWTForward(J=num_dwt_levels, wave='haar', mode='zero').to(device)
    ll, h = dwt(latent)
    return ll, h

#################################################################################
#                                  Training Loop                                #
#################################################################################

# Replace the distributed setup section with this simple fix:

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Force single GPU mode - no distributed processing
    distributed = False
    world_size = 1
    rank = 0
    device = 0  # Use GPU 0
    seed = args.global_seed
    num_dwt_levels = args.num_dwt_levels

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting single GPU mode: rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup feature folder
    feature_type = 'latent' if args.use_latent else 'image'
    os.makedirs(args.features_path, exist_ok=True)
    os.makedirs(os.path.join(args.features_path, f'imagenet256/train/{feature_type}_{num_dwt_levels}_dwt_LL'), exist_ok=True)
    os.makedirs(os.path.join(args.features_path, f'imagenet256/train/{feature_type}_{num_dwt_levels}_dwt_highfreq'), exist_ok=True)
    os.makedirs(os.path.join(args.features_path, f'imagenet256/val/{feature_type}_{num_dwt_levels}_dwt_LL'), exist_ok=True)
    os.makedirs(os.path.join(args.features_path, f'imagenet256/val/{feature_type}_{num_dwt_levels}_dwt_highfreq'), exist_ok=True)

    # Create model
    if args.use_latent:
        assert args.image_size % 8 == 0, "Image size must be divisible by 8."
        # latent_size = args.image_size // 8
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    train_dataset = CustomDataset(args.data_path, train_ratio=0.7, split='train', image_size=args.image_size)
    val_dataset = CustomDataset(args.data_path, train_ratio=0.7, split='val', image_size=args.image_size)

    # Simple DataLoader without distributed sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,  # Use full batch size since no distribution
        shuffle=True,  # Enable shuffle for better class distribution
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Rest of your training loop remains the same
    train_steps = 0
    for x in train_loader:
        x = x.to(device)
        # y = y.to(device)
        with torch.no_grad():
            if args.use_latent:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x, h = extract_dwt_features(x, num_dwt_levels=num_dwt_levels, device=device)
        x = x.detach().cpu().numpy()  # (B, C, H/2^n, W/2^n)
        h = [level.detach().cpu().numpy() for level in h]   # list [(B, C, 3, H/2^n, W/2^n)] of length num_dwt_levels
        y = h[-1]  # Use only the highest frequency components for labels
        
        for i in range(x.shape[0]):
            np.save(f'{args.features_path}/imagenet256/train/{feature_type}_{num_dwt_levels}_dwt_LL/{train_steps}.npy', x[i:i+1])
            np.save(f'{args.features_path}/imagenet256/train/{feature_type}_{num_dwt_levels}_dwt_highfreq/{train_steps}.npy', y[i:i+1])
            # print(y[i:i+1])
            train_steps += 1
        
        if train_steps % 200 == 0:
            print(f"Processed {train_steps} training samples")

    val_steps = 0
    for x in val_loader:
        x = x.to(device)
        # y = y.to(device)
        with torch.no_grad():
            if args.use_latent:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x, h = extract_dwt_features(x, num_dwt_levels=num_dwt_levels, device=device)
        x = x.detach().cpu().numpy()
        h = [level.detach().cpu().numpy() for level in h]
        
        for i in range(x.shape[0]):
            np.save(f'{args.features_path}/imagenet256/val/{feature_type}_{num_dwt_levels}_dwt_LL/{val_steps}.npy', x[i:i+1])
            np.save(f'{args.features_path}/imagenet256/val/{feature_type}_{num_dwt_levels}_dwt_highfreq/{val_steps}.npy', y[i:i+1])
            # print(y[i:i+1])
            val_steps += 1
        
        if val_steps % 200 == 0:
            print(f"Processed {val_steps} val samples")


    # Save zip files
    import zipfile
    with zipfile.ZipFile(f'{args.features_path}/imagenet256/train/{feature_type}_{num_dwt_levels}_dwt_LL.zip', 'w') as zipf:
        for file in glob(f'{args.features_path}/imagenet256/train/{feature_type}_{num_dwt_levels}_dwt_LL/*.npy'):
            zipf.write(file, os.path.relpath(file, args.features_path))
    with zipfile.ZipFile(f'{args.features_path}/imagenet256/train/{feature_type}_{num_dwt_levels}_dwt_highfreq.zip', 'w') as zipf:
        for file in glob(f'{args.features_path}/imagenet256/train/{feature_type}_{num_dwt_levels}_dwt_highfreq/*.npy'):
            zipf.write(file, os.path.relpath(file, args.features_path))
    with zipfile.ZipFile(f'{args.features_path}/imagenet256/val/{feature_type}_{num_dwt_levels}_dwt_LL.zip', 'w') as zipf:
        for file in glob(f'{args.features_path}/imagenet256/val/{feature_type}_{num_dwt_levels}_dwt_LL/*.npy'):
            zipf.write(file, os.path.relpath(file, args.features_path))
    with zipfile.ZipFile(f'{args.features_path}/imagenet256/val/{feature_type}_{num_dwt_levels}_dwt_highfreq.zip', 'w') as zipf:
        for file in glob(f'{args.features_path}/imagenet256/val/{feature_type}_{num_dwt_levels}_dwt_highfreq/*.npy'):
            zipf.write(file, os.path.relpath(file, args.features_path))
    
    print(f"Finished processing {train_steps} val samples.")
    
    


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=10)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--num-dwt-levels", type=int, default=1, help="Number of DWT levels to use for feature extraction.")
    parser.add_argument("--use-latent", type=bool, default=False, help="Whether to extract features from VAE latent space or directly from images.")
    args = parser.parse_args()
    main(args)