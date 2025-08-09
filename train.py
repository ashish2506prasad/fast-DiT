# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
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


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs("training_image_generation", exist_ok=True)

    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        # Set default values for non-main processes
        experiment_dir = f"{args.results_dir}/000-{args.model.replace('/', '-')}"

    # Move this outside the if block so all processes can access it:
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    if args.num_dwt_levels is None:
        latent_size = args.image_size // 8
    else:
        latent_size = args.image_size // (8 * (2 ** args.num_dwt_levels))

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    if args.num_dwt_levels is None:
        features_dir = f"{args.feature_path}/imagenet256_features/imagenet256_features"
        labels_dir = f"{args.feature_path}/imagenet256_labels/imagenet256_labels"
    else:
        features_dir = f"{args.feature_path}/imagenet256_{args.num_dwt_levels}_dwt_features/imagenet256_{args.num_dwt_levels}_dwt_features"
        labels_dir = f"{args.feature_path}/imagenet256_{args.num_dwt_levels}_dwt_labels/imagenet256_{args.num_dwt_levels}_dwt_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
        print(f"Training for {args.epochs} epochs...")
    loss_list = []
    epoch = 0
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
            print(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                loss_list.append(avg_loss)
                start_time = time()

            if args.save_img_after > 0:
                if train_steps%(args.save_img_after)==0:
                    with torch.no_grad():
                        model.eval()
                        torch.manual_seed(0)
                        torch.set_grad_enabled(False)
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        # assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
                        assert args.image_size in [256, 512]
                        # assert args.num_classes == 1000
                        num_sampling_steps=200
                        diffusion = create_diffusion(str(num_sampling_steps))
                        print("created diffusion")
                        vae_ = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
                        # Labels to condition the model with (feel free to change):
                        for class_label in range(1, 19, 3):
                            class_labels = [class_label]  # Change this to sample different classes
                            print(f"Sampling images for class {class_labels[0]} at step {train_steps}")

                            n = len(class_labels)
                            z = torch.randn(n, 4, latent_size, latent_size, device=device)
                            y = torch.tensor(class_labels, device=device)

                            # Setup classifier-free guidance:
                            z = torch.cat([z, z], 0)
                            y_null = torch.tensor([args.num_classes] * n, device=device)
                            y = torch.cat([y, y_null], 0)
                            model_kwargs_ = dict(y=y)

                            # Sample images:
                            samples = diffusion.p_sample_loop(
                                model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs_, progress=True, device=device,
                                save_timestep_output=args.save_timestep_images, class_gen=class_labels[0], train_step=train_steps
                            )
                            print("sampled images")
                            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                            print(samples.shape)
                            if args.num_dwt_levels is not None:
                                # perform IDWT, then divide by 0.18215 and decode using VAE
                                print("performing IDWT")
                                idwt = DWTInverse(wave='haar', mode='zero').to(device)
                                for _ in range(args.num_dwt_levels):
                                    dummy_high_frequency = torch.zeros(
                                                                        samples.shape[0], samples.shape[1], 3, 
                                                                        samples.shape[2], samples.shape[3], 
                                                                        device=samples.device  # Add this line to ensure same device
                                                                    )
                                    samples = idwt((samples, [dummy_high_frequency]))
                                    
                            samples = vae_.decode(samples / 0.18215).sample
                            # Save and display images:
                            # save image like 000001 etc in 7 digit numbers
                            save_image(samples, f"training_image_generation/class_{class_label}/sample_{class_labels[0]}_{train_steps:8d}.png", nrow=4, normalize=True, value_range=(-1, 1))

                        import zipfile
                        with zipfile.ZipFile('training_image_generation.zip', 'w') as zipf:
                            for file in glob(f'training_image_generation/*.png'):
                                zipf.write(file, os.path.relpath(file, args.features_path))
                model.train()
                torch.set_grad_enabled(True)

    with open(f"./results/loss.json", "w") as f:
        json.dump(loss_list, f)

    if accelerator.is_main_process:
        checkpoint = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                }
        checkpoint_path = f"{checkpoint_dir}/dit_xs_2 _epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        print(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=700)
    parser.add_argument("--token-mixer", type=str, default="softmax", choices=["linformer", "nystromformer", "performer", "softmax"])
    parser.add_argument("--save-img-after", type=int, default=50)  # set to -1 to disable image saving during training
    parser.add_argument("--save-timestep-images", type=False, default=False, help="Save images at each timestep during sampling.")
    parser.add_argument("--num-dwt-levels", type=int, default=None, help="Number of DWT levels to use for feature extraction.")
    args = parser.parse_args()

    import os
    import zipfile

    def zip_folder(folder_path, output_path):
        # Create a ZipFile object in write mode
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Store relative path to preserve folder structure
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)

    # Example usage    
    # def get_latest_checkpoint(path='.'):
    #     ckpts = [f for f in os.listdir(path) if f.startswith('checkpoint_step_')]
    #     if not ckpts:
    #         return None
    #     ckpts = sorted(ckpts, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    #     return os.path.join(path, ckpts[-1])

    main(args)

    # save the zip of resulter folder and debug_outputs folder
    zip_folder("./results", "./results.zip")
    # zip_folder("./debug_outputs", "./debug_outputs.zip")