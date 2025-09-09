import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
from accelerate import Accelerator
import json
from torchvision.utils import save_image
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from sample import sample_main
from pytorch_wavelets import DWTForward, DWTInverse
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


# Create custom transform classes to avoid lambda functions
class CenterCropTransform:
    def __init__(self, image_size):
        self.image_size = image_size
    
    def __call__(self, pil_image):
        return center_crop_arr(pil_image, self.image_size)

class IdentityTransform:
    def __call__(self, x):
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = IdentityTransform()
    
    def forward(self, x):
        return self.block(x) + self.residual(x)

class WaveletDecoder(nn.Module):
    def __init__(self, dropout=0.5, in_chanel=3, res_depth=5):  
        super().__init__()
        self.in_channel = in_chanel  
        
        # Each resblock has its own fc layer for alpha/beta
        fc_layer = nn.Sequential(
            nn.Linear(1, 8), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(8, 2), 
            nn.ReLU()
        )
        self.fc1 = nn.ModuleList([deepcopy(fc_layer) for _ in range(res_depth)])
        
        # Initial conv stem
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chanel, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Residual blocks
        self.resblocks = nn.ModuleList([
            ResidualBlock(32, 32, dropout=dropout) for _ in range(res_depth)
        ])
        
        # Final convs
        self.cnn2 = nn.Sequential(
            nn.BatchNorm2d(32),  
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=16, out_channels=in_chanel*3, kernel_size=3, padding=1)
        )

    def forward(self, x, num_dwt_levels):
        """
        x: Tensor of shape (B, C, H, W)
        num_dwt_levels: Tensor of shape (B, 1)
        """
        x = self.cnn1(x)

        # Apply each residual block with its alpha/beta scaling
        for fc, resblock in zip(self.fc1, self.resblocks):
            x = resblock(x)
            params = fc(num_dwt_levels)  # Shape: (B, 2)
            alpha = params[:, 0:1].view(-1, 1, 1, 1)  # (B,1,1,1)
            beta  = params[:, 1:2].view(-1, 1, 1, 1)  # (B,1,1,1)
            x = alpha * x + beta

        x = self.cnn2(x)

        # Output shape: (B, C, 3, H, W)
        x = x.view(x.size(0), self.in_channel, 3, x.size(2), x.size(3))
        return x
    
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
    if dist.is_initialized() and dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger or single GPU mode
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    return logger

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

def extract_dwt_features(latent, num_dwt_levels=1, device='cpu'):
    """
    Extract features from the latent representation using Discrete Wavelet Transform (DWT).
    Performs n levels of DWT on the latent representation. So the output size will be 2^n times smaller than the input size.
    Args:
        latent (torch.Tensor): Latent representation of shape (batch_size, channels, height, width).
        num_dwt_levels (int): Number of DWT levels to apply.
        device (str): Device to perform the computation on ('cpu' or 'cuda').
    Returns:
        torch.Tensor: DWT features of shape (batch_size, channels, height // (2 ** num_dwt_levels), width // (2 ** num_dwt_levels)).
    """
    dwt = DWTForward(J=num_dwt_levels, wave='haar', mode='zero').to(device)
    ll, x = dwt(latent)
    # x.shape: (B, n, 3, C, H/2^n, W/2^n)  # 3 corresponds to (LH, HL, HH)
    return ll, x
    
class CustomDataset(Dataset):
    """
    Custom dataset to load images from a directory structure.
    Expects images to be in subdirectories of the parent directory.
    Args:
        parent_dir (str): Path to the parent directory containing image subdirectories.
    Returns
        torch.Tensor: Transformed image tensor.
        does not return class as they are not needed for feature extraction
    """
    def __init__(self, parent_dir, test_data=None, split='train', feature_type ='image', num_dwt_levels = 1):
        self.parent_dir = parent_dir
        # Fixed: Use parent_dir parameter instead of hardcoded path
        if split in ['train', 'val']:
            self.image_paths = sorted(glob(f"{parent_dir}/{split}/{feature_type}_{num_dwt_levels}_dwt_LL/*.npy"))
            self.label_paths = sorted(glob(f"{parent_dir}/{split}/{feature_type}_{num_dwt_levels}_dwt_highfreq/*.npy"))
            print(f"Found {len(self.image_paths)} images in {split} set")
        else:
            classes = os.listdir(test_data)
            self.image_paths = []
            for class_ in classes:
                class_path = os.path.join(test_data, class_)
                sorted_images = sorted(glob(f"{class_path}/*/*.[Jj][Pp][Ee][Gg]"))  # matches .JPEG/.jpeg
                print(f"Found {len(sorted_images)} images in class {class_}")
                self.image_paths += sorted_images[int(len(sorted_images)*0.8):]  # Use 20% of images for testing
            print(f"Found {len(self.image_paths)} images in {split} set ")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx] if hasattr(self, 'label_paths') else None
        
        # Check if this is a test split with image files or train/val split with .npy files
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load image file for test data
            image = Image.open(img_path).convert('RGB')
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Adjust size as needed
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
            image = transform(image)
        else:
            # Load .npy file for train/val data
            image = np.load(img_path)
            image = torch.from_numpy(image).float()
        
        if label_path:
            label = np.load(label_path)
            label = torch.from_numpy(label).float()
            return image, label
        else:
            # Return a dummy tensor for test data where no labels exist
            dummy_label = torch.zeros(1)  # Placeholder tensor
            return image, dummy_label

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Add before saving JSON:
    feature_type = 'latent' if args.use_latent else 'image'
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/{feature_type}", exist_ok=True)

    if args.where == "local":
        # Local single GPU setup
        distributed = False
        rank = 0
        world_size = 1
        device = 0
        torch.cuda.set_device(device)
        print("Running in local single GPU mode")
    else:
        # Distributed training setup
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank
        torch.cuda.set_device(device)
        distributed = True
        print(f"Running in distributed mode: rank={rank}, world_size={world_size}")

    num_dwt_levels = args.num_dwt_levels

    loss_list = []

    # Create model
    if args.use_latent:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Fixed: Pass image_size parameter to CustomDataset
    train_dataset = CustomDataset(args.data_path, split='train', feature_type=feature_type, num_dwt_levels=num_dwt_levels) 
    val_dataset = CustomDataset(args.data_path, split='val', feature_type=feature_type, num_dwt_levels=num_dwt_levels)
    shuffle = True if args.where == "local" else None

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if args.where == "cluster" else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if args.where == "cluster" else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,  # Use full batch size since no distribution
        shuffle=shuffle,  # Enable shuffle for better class distribution
        sampler=train_sampler,
        num_workers=args.num_workers,  
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.global_batch_size,  # Use full batch size since no distribution
        shuffle=shuffle,  # Enable shuffle for better class distribution
        sampler=val_sampler,
        num_workers=args.num_workers,  
        pin_memory=True,
        drop_last=False
    )
    
    # Fixed: Use proper in_chanel parameter
    wavelet_docoder_model = WaveletDecoder(in_chanel=4 if args.use_latent else 3, res_depth=5).to(device)
    if args.where == "cluster":
        wavelet_docoder_model = DDP(wavelet_docoder_model, device_ids=[device])
    optimizer = torch.optim.AdamW(wavelet_docoder_model.parameters(), lr=1e-4, weight_decay=1e-2)
    logger = create_logger(args.results_dir)
    
    for epoch in range(args.epochs):
        if args.where == "cluster":
            train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(train_loader):
            optimizer.zero_grad()
            start_time = time()
            ll = img.to(device)
            ll.squeeze_(1)  # (B, 1, C, H, W) -> (B, C, H, W)
            label = label.to(device)
            label.squeeze_(1)  # (B, 1, C, 3, H, W) -> (B, C, 3, H, W)
            if epoch == 0 and step == 0 and rank == 0:
                print(f"Input LL shape: {ll.shape}, label shape: {label.shape}")

            # ll.shape = (B, C, H/2^n, W/2^n)
            # label.shape = (B, C, 3, H/2^n, W/2^n)  # 3 corresponds to (LH, HL, HH)
             
            num_dwt_levels_tensor = torch.ones(ll.shape[0], 1, device=device, dtype=torch.float32) * num_dwt_levels     # batch size, 1
            output = wavelet_docoder_model(ll, num_dwt_levels_tensor)
            loss = nn.MSELoss()(output, label)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if step % args.log_every == 0:
                logger.info(f"Epoch {epoch}, step {step}, loss {loss.item():.4f}, time {time() - start_time:.2f}s")
                print(f"Epoch {epoch}, step {step}, loss {loss.item():.4f}, time {time() - start_time:.2f}s")
        
        if epoch % args.ckpt_every == 0 and (args.where=='local' or rank == 0):
            # Fixed: Save model state dict properly without DDP wrapper
            val_loss = 0.0
            if args.where == "cluster":
                val_sampler.set_epoch(epoch)
            with torch.no_grad():
                wavelet_docoder_model.eval()
                for ll, label in val_loader:
                    ll = ll.to(device)  
                    ll.squeeze_(1)  # (B, 1, C, H, W) -> (B, C, H, W)
                    print("Validation LL shape:", ll.shape)
                    label = label.to(device)
                    label.squeeze_(1)  # (B, 1, C, 3, H, W) -> (B, C, 3, H, W)
                    num_dwt_levels_tensor = torch.ones(ll.shape[0], 1, device=device, dtype=torch.float32) * num_dwt_levels
                    output = wavelet_docoder_model(ll, num_dwt_levels_tensor)
                    val_loss += nn.MSELoss()(output, label).item()
                val_loss /= len(val_loader)  # Divide by number of batches, not dataset length
                wavelet_docoder_model.train()  # Switch back to training mode
            logger.info(f"Validation loss: {val_loss:.4f}")
            print(f"Validation loss: {val_loss:.4f}")
            if args.where == "local":
                # os.makedirs(os.path.dirname(f"{args.results_dir}/{feature_type}"), exist_ok=True)
                torch.save(wavelet_docoder_model.state_dict(), f"{args.results_dir}/{feature_type}/wavelet_decoder_epoch_{epoch}.pth")
            else:
                # os.makedirs(os.path.dirname(f"{args.results_dir}/{feature_type}"), exist_ok=True)
                torch.save(wavelet_docoder_model.module.state_dict(), f"{args.results_dir}/{feature_type}/wavelet_decoder_epoch_{epoch}.pth")
            logger.info(f"Saved checkpoint at epoch {epoch}")
            print(f"Saved checkpoint at epoch {epoch} ")
    
    # Fixed: Save final model without DDP wrapper
    if (args.where=='local' or rank == 0):
        if args.where == "local":
            torch.save(wavelet_docoder_model.state_dict(), f"{args.results_dir}/{feature_type}/wavelet_decoder_{num_dwt_levels}_final.pth")
        else:   
            torch.save(wavelet_docoder_model.module.state_dict(), f"{args.results_dir}/{feature_type}/wavelet_decoder_{num_dwt_levels}_final.pth")
        # os.makedirs(f"{args.results_dir}/{feature_type}/wavelet_decoder_loss_{num_dwt_levels}.json", exist_ok=True)
        with open(f"{args.results_dir}/{feature_type}/wavelet_decoder_loss_{num_dwt_levels}.json", 'w') as f:
            json.dump(loss_list, f)

        logger.info("Training complete.")

def eval(args):
    device = 0
    num_dwt_levels = args.num_dwt_levels
    wavelet_docoder_model = WaveletDecoder(in_chanel=4 if args.use_latent else 3, res_depth=5).to(device)
    # Fixed: Load state dict without DDP wrapper expectation
    feature_type = 'latent' if args.use_latent else 'image'
    wavelet_docoder_model.load_state_dict(torch.load(f"{args.results_dir}/{feature_type}/wavelet_decoder_{num_dwt_levels}_final.pth", map_location='cpu'))
    wavelet_docoder_model.eval()

    if args.use_latent:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    if args.ood_eval:
        dataset = CustomDataset(args.ood_eval, split='test', test_data=args.ood_eval) 
    else:
        dataset = CustomDataset(args.data_path, split='test', test_data=args.test_data)

    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,  # Evaluate one image at a time
        shuffle=False,
        num_workers=0,  
        pin_memory=True,
        drop_last=False
    )
    os.makedirs(os.path.join(args.results_dir, f'/{feature_type}/eval_results_{num_dwt_levels}_dwt'), exist_ok=True)

    with torch.no_grad():
        error_list = []
        final_reconstruction_error = []
        for idx, (image, _) in enumerate(loader):
            latent = image.to(device)

            if args.use_latent:
                with torch.no_grad():
                    latent = vae.encode(image.to(device)).latent_dist.sample() * 0.18215

            ll, high_freq_features = extract_dwt_features(latent, num_dwt_levels=num_dwt_levels, device=device)
            # high_freq_features.pop()
            # high_freq_features.append(high_freq)

            num_dwt_levels_tensor = torch.ones(1, 1, device=device, dtype=torch.float32) * num_dwt_levels
            output = wavelet_docoder_model(ll, num_dwt_levels_tensor)
            # print("***********chkpt 1", output.shape, high_freq_features[-1].shape)
            error = nn.MSELoss()(output, high_freq_features[-1])
            error_list.append(error.item())
            high_freq_features.pop()
            
            # Reconstruct the image using the predicted high-frequency components
            for i in range(num_dwt_levels, 0, -1):
                if i == num_dwt_levels:
                    # lh_pred, hl_pred, hh_pred = output[:, :, 0], output[:, :, 1], output[:, :, 2]
                    ll = DWTInverse(wave='haar', mode='zero').to(device)((ll, [output]))
                    # print("***********chkpt 2",ll.shape)
                else:
                    print(i)
                    # print("***********chkpt 3",ll.shape)
                    # print("***********chkpt 4", type(high_freq_features[-1]), high_freq_features[-1].shape) 
                    ll = DWTInverse(wave='haar', mode='zero').to(device)((ll, [high_freq_features[-1]]))
                    high_freq_features.pop()
            
            # Final reconstruction
            # print("***********chkpt 5",ll.shape, latent.shape)
            final_reconstruction_error.append(nn.MSELoss()(ll, latent).item())
            print(f"Image {idx}, DWT Levels {num_dwt_levels}, High-Freq MSE: {error.item():.6f}, Final Recon MSE: {final_reconstruction_error[-1]:.6f}")
            
            # save the final reconstruction
            if args.use_latent:
                reconstructed_image = vae.decode(ll / 0.18215).sample
                reconstructed_image = (reconstructed_image / 2 + 0.5).clamp(0, 1)
            else:
                reconstructed_image = (ll / 2 + 0.5).clamp(0, 1)
            os.makedirs(f"{args.results_dir}/{feature_type}/eval_results_{num_dwt_levels}_dwt", exist_ok=True)
            save_image(reconstructed_image, os.path.join(args.results_dir, f"{feature_type}", f'eval_results_{num_dwt_levels}_dwt', f'final_reconstructed_{idx}.png'))
            if idx % 100 == 0:
                print(f"Saved {idx} images")



if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    # parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--num-dwt-levels", type=int, default=1, help="Number of DWT levels to use for feature extraction.")
    parser.add_argument("--use-latent", type=bool, default=False, help="Use VAE latent space")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--ood-eval", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--res-depth", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--where", type=str, choices=["local", "cluster"], default="cluster", help="Run locally or on cluster")
    args = parser.parse_args()
    
    main(args)
    if args.where == "local" or dist.get_rank() == 0:
        eval(args)
        
        # Create zip file
        # import zipfile
        # import shutil
        # def create_zip_archive(source_dir, output_filename):
        #     shutil.make_archive(output_filename.replace('.zip', ''), 'zip', source_dir)
        
        # zip_filename = f"{args.results_dir}_training_results.zip"
        # create_zip_archive(args.results_dir, zip_filename)
        # print(f"Results saved and zipped to: {zip_filename}")
    
    if args.where != "local":
        cleanup()  