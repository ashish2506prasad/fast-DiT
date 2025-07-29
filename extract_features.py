# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


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
    dwt = DWTForward(J=1, wave='haar').to(device)
    ll = latent
    for _ in range(num_dwt_levels):
        ll, (lh, hl, hh) = dwt(ll)
        ll = ll.to(device)
        lh = lh.to(device)
        hl = hl.to(device)
        hh = hh.to(device)
    
    return ll


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP if launched with torch.distributed.launch or torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        distributed = True
    else:
        distributed = False

    num_dwt_levels = args.num_dwt_levels
    assert num_dwt_levels >= 0, "Number of DWT levels must be non-negative."
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup feature folder only from rank 0
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'imagenet256_{num_dwt_levels}_dwt_features'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'imagenet256_{num_dwt_levels}_dwt_labels'), exist_ok=True)

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.global_seed)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // world_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_steps = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            x = extract_dwt_features(x, num_dwt_levels=num_dwt_levels, device=device)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        # Save each sample in the batch individually to maintain file structure
        for i in range(x.shape[0]): # x.shape = (batch_size, channels, height, width)
            np.save(f'{args.features_path}/imagenet256_{num_dwt_levels}_dwt_features/{train_steps}.npy', x[i:i+1])
            np.save(f'{args.features_path}/imagenet256_{num_dwt_levels}_dwt_labels/{train_steps}.npy', y[i:i+1])
            train_steps += 1
        
        if train_steps % 100 == 0:  # Print less frequently
            print(f"Processed {train_steps} samples")

    # save a zip file of the features and labels
    if rank == 0:
        import zipfile
        with zipfile.ZipFile(f'{args.features_path}/imagenet256_{num_dwt_levels}_dwt_features.zip', 'w') as zipf:
            for file in glob(f'{args.features_path}/imagenet256_{num_dwt_levels}_dwt_features/*.npy'):
                zipf.write(file, os.path.relpath(file, args.features_path))
        with zipfile.ZipFile(f'{args.features_path}/imagenet256_{num_dwt_levels}_dwt_labels.zip', 'w') as zipf:
            for file in glob(f'{args.features_path}/imagenet256_{num_dwt_levels}_dwt_labels/*.npy'):
                zipf.write(file, os.path.relpath(file, args.features_path))
    print(f"Finished processing {train_steps} samples.")
    
    


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=10)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--num-dwt-levels", type=int, default=1, help="Number of DWT levels to use for feature extraction.")
    args = parser.parse_args()
    main(args)
