import os
import random
import torch

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import transforms
from optim_utils import set_random_seed

import numpy as np

def visualize_tensor(tensor, name='./tmp.png'):
    channels = tensor.squeeze(0).cpu().numpy()
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    for i, ax in enumerate(axs):
        ax.imshow(channels[i], cmap='gray')
        ax.set_title(f'Channel {i+1}')
        ax.axis('off')

    plt.savefig(name)
    plt.close()


def tensor_to_pil(images):
    # Move post-processing in decode to here.
    images = (images / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()

    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def display_image(image, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()

def calculate_metric(reversed_latents, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement:
        reversed_latents_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_fft = reversed_latents
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        metric = torch.abs(reversed_latents_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return metric

def distort_image(img, seed, args):
    if args.r_degree is not None:
        img = transforms.RandomRotation((args.r_degree, args.r_degree))(img)

    if args.jpeg_ratio is not None:
        img.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img = transforms.RandomResizedCrop(img.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img)
        
    if args.gaussian_blur_r is not None:
        img = img.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img = Image.fromarray(np.clip(np.array(img) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img = transforms.ColorJitter(brightness=args.brightness_factor)(img)

    return img

def show_images_pair(img1, img2, title, save_path=None):
    # Convert PIL images to NumPy arrays
    img1_np = np.array(img1).astype(np.float32)
    img2_np = np.array(img2).astype(np.float32)
    
    # Calculate the absolute difference between the two images
    difference = np.abs(img1_np - img2_np)
    
    # Normalize the difference to [0, 1]
    diff_min = difference.min()
    diff_max = difference.max()
    if diff_max - diff_min > 0:
        norm_diff = (difference - diff_min) / (diff_max - diff_min)
    else:
        norm_diff = difference  # if image is constant
    
    # norm_diff = norm_diff / 2 + 0.5
    norm_diff = np.clip((norm_diff - 0.5) * 1.5 + 0.5, 0, 1)
    # Optionally save the difference image
    if save_path is not None:
        plt.imsave(save_path, norm_diff, cmap='gray')
    
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display the first image
    axs[0].imshow(img1)
    axs[0].axis('off')
    axs[0].set_title("Image 1")
    
    # Display the second image
    axs[1].imshow(img2)
    axs[1].axis('off')
    axs[1].set_title("Image 2")
    
    # Display the normalized difference image in grayscale
    axs[2].imshow(norm_diff, cmap='gray')
    axs[2].axis('off')
    axs[2].set_title("Difference (B/W)")
    
    # Add the main title and adjust layout
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()