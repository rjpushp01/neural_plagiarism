import argparse
import copy
import os
import warnings
import glob
from pathlib import Path
import requests

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from statistics import mean, stdev

from pycocotools.coco import COCO
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL

import numpy as np
import PIL.Image
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from attack_stable_diffusion import AttackStableDiffusionPipeline
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from optim_utils import *
from io_utils import *
from log import Log
# from watermarker import InvisibleWatermarker
from utils.image_processing import tensor_to_pil

# Ignore warnings for cleaner output.
warnings.filterwarnings("ignore")

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Optimize latent images with watermarking")
parser.add_argument('--target_folder', required=True, help='Folder containing target images')
parser.add_argument('--start', type=int, default=0, help='Starting index for processing images')
parser.add_argument('--end', type=int, default=10, help='Ending index for processing images')
parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
parser.add_argument('--image_length', type=int, default=512, help='Length of the image (square assumed)')
parser.add_argument('--model_id', default='Manojb/stable-diffusion-2-1-base', help='Model ID for the diffusion pipeline')
parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate per prompt')
parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for stable diffusion')
parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of diffusion steps for inference')
parser.add_argument('--attack_num_inference_steps', type=int, default=50, help='Number of steps for reverse diffusion')
parser.add_argument('--output_folder', default='./outputs/', help='Folder for saving output images and logs')
parser.add_argument('--start_step', type=int, default=0, help='Starting step for optimization.')
parser.add_argument('--shortcut_step', type=int, default=-1, help='Build a shortcut from this step to step 0.')
parser.add_argument('--iters', type=int, default=10, help='Number of optimization iterations')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimization')
parser.add_argument('--gamma1', type=float, default=0.1, help='Weight for latent difference')
parser.add_argument('--gamma2', type=float, default=1e5, help='Weight for semantic difference')
parser.add_argument('--gamma3', type=float, default=1e-3, help='Weight for image difference')
parser.add_argument('--eps', nargs='+', type=float, default=[10, 15], help='Selected bounds')
parser.add_argument('--k', nargs='+', type=int, default=[25, 45], help='Selected timesteps')
parser.add_argument('--watermark_text', type=str, default='test', help='Watermark key text')
parser.add_argument('--watermark_method', default='dwtDctSvd', help='Watermarking method to use (e.g., "dwtDctSvd", "rivaGan")')
parser.add_argument('--gen_seed', type=int, default=0, help='Seed for random generation of images')
parser.add_argument('--decode_inv', action='store_true', help='Learn the VAE encoding by regression')
args = parser.parse_args()

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        # Ensure the values remain in the [0, 1] range
        return torch.clamp(noisy_tensor, 0., 1.)

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0

def get_bit_acc_success(decode_text, expected_text='test', threshold=24/32, is_bitstream=False):
    if is_bitstream:
        diff = [decode_text[i] != expected_text[i] for i in range(len(expected_text))]
        bit_acc = 1 - sum(diff) / len(diff)
        success = bit_acc > threshold
    else:
        expected_bits = bytearray_to_bits(expected_text.encode('utf-8'))

        if isinstance(decode_text, bytes):
            wm_bits = bytearray_to_bits(decode_text)
        elif isinstance(decode_text, str):
            wm_bits = bytearray_to_bits(decode_text.encode('utf-8'))
        else:
            raise TypeError("decode_text must be of type bytes or str.")

        bit_acc = (np.array(expected_bits) == np.array(wm_bits)).mean()
        success = bit_acc > threshold
    return bit_acc, success


# Ensure output directories exist.
os.makedirs(args.output_folder, exist_ok=True)
watermarked_folder = os.path.join(args.output_folder, 'watermarked')
os.makedirs(watermarked_folder, exist_ok=True)
reversed_folder = os.path.join(args.output_folder, "reversed")
os.makedirs(reversed_folder, exist_ok=True)

# Set up logging.
log = Log("{}/log.txt".format(args.output_folder))
log.info("Arguments: {}".format(vars(args)))

# Set device.
device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"

# Load the diffusion model.
scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
# scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
pipe = AttackStableDiffusionPipeline.from_pretrained(
    args.model_id,
    scheduler=scheduler,
    vae=vae,
    torch_dtype=torch.float16,
).to(device)

set_random_seed(args.gen_seed)
pipe.enable_attention_slicing(1)
# Generate the watermark and save it to a file for later detection.
# if args.watermark_text:
#     log.info("Watermark text: {}".format(args.watermark_text))
#     watermark = args.watermark_text.encode('utf-8')
# else:
#     watermark = generate_watermark().encode('utf-8')

# watermark_file = Path(args.output_folder) / "watermark.txt"
# with open(watermark_file, "w") as f:
#     f.write(watermark.decode('utf-8'))

# Initialize the invisible watermarker.
# wmarker = InvisibleWatermarker(args.watermark_text, args.watermark_method)

# Lists to hold watermark detection accuracies.
# original_accuracies = []
# reversed_accuracies = []
# attack_accuracies = []

# Get list of original image paths.
ori_img_paths = glob.glob(os.path.join(args.target_folder, '*.*'))
ori_img_paths = sorted([path for path in ori_img_paths if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
ori_img_paths = ori_img_paths[args.start:args.end]
log.info("Total images {}".format(len(ori_img_paths)))

# Process images.
for i, ori_img_path in enumerate(ori_img_paths):
    

    img_name = os.path.basename(ori_img_path)
    
    # --- Original Image: Embed and Decode Watermark ---
    # wm_img_path = os.path.join(watermarked_folder, img_name)
    # wmarker.encode(ori_img_path, wm_img_path)
    
    # decoded_wm_original = wmarker.decode(wm_img_path)
    # wm_bit_acc_original, wm_success_original = get_bit_acc_success(
    #     decoded_wm_original,
    #     expected_text=args.watermark_text,
    #     threshold=24/32
    # )
    # log.info("Original Image {:04d} - Decoded Watermark: {} | Bit Accuracy: {:.2f}% (Success: {})".format(
    #     i, decoded_wm_original, wm_bit_acc_original * 100, wm_success_original))
    # original_accuracies.append(wm_bit_acc_original)
    
    # --- Attack Pipeline Setup ---
    torch.cuda.empty_cache()  # clear cached allocations from previous image
    empty_prompt = ""
    empty_embedding = pipe.encode_prompt(
        empty_prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    ).detach()

    target_img_pil = Image.open(ori_img_path)
    target_img = transform_img(target_img_pil, target_size=args.image_length).unsqueeze(0).to(empty_embedding.dtype).to(device).detach()
    
    # Set timesteps for reverse diffusion.
    pipe.scheduler.set_timesteps(args.attack_num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    alphas_cumprod = pipe.scheduler.alphas_cumprod
    log.info("Timesteps ({}): {}".format(len(timesteps), timesteps))
    
    for j in range(args.num_images):
        seed = 10*j + i + args.gen_seed
        set_random_seed(seed)
        # Decode latents using inversion if specified.
        if args.decode_inv:
            log.info("Inversing target latents.")
            target_latents = pipe.decoder_inv(target_img).detach()
        else:
            target_latents = pipe.get_image_latents(target_img, sample=False).detach()
        
        # --- Forward Diffusion: Collect Anchor Latents ---
        anchor_latents = []
        def collect_latents(step, timestep, latents):
            anchor_latents.append(latents.clone().detach())

        text_embeddings = pipe.get_text_embedding(empty_prompt).detach()
        _ = pipe.forward_diffusion(
            latents=target_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.attack_num_inference_steps,
            callback=collect_latents,
            callback_steps=1
        )

        # Reverse the collected anchor latents.
        anchor_latents = list(reversed(anchor_latents))

        with torch.no_grad():
            outputs_reversed = pipe(
                empty_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.attack_num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                output_type="latent",
                head_start_step=args.start_step,
                head_start_latents=anchor_latents[args.start_step]
            )
            reversed_latents = outputs_reversed.images.detach()
            decoded_reversed_image = pipe.decode_latents(reversed_latents)
        reversed_image_pil = tensor_to_pil(decoded_reversed_image)[0]
        reversed_image_path = os.path.join(reversed_folder, f"image_{i:04d}.png")
        reversed_image_pil.save(reversed_image_path)
        
        # --- Reversed Image: Decode Watermark ---
        # decoded_wm_reversed = wmarker.decode(reversed_image_path)
        # wm_bit_acc_reversed, wm_success_reversed = get_bit_acc_success(
        #     decoded_wm_reversed,
        #     expected_text=args.watermark_text,
        #     threshold=24/32
        # )
        # log.info("Reversed Image {:04d} - Decoded Watermark: {} | Bit Accuracy: {:.2f}% (Success: {})".format(
        #     i, decoded_wm_reversed, wm_bit_acc_reversed * 100, wm_success_reversed))
        # reversed_accuracies.append(wm_bit_acc_reversed)
        
        # --- Attack: Optimize Latents ---
        dist_func = lambda x, y: torch.norm(x - y, p=2)
        align_func = lambda x, y, mu, std: mu * dist_func(x.mean(dim=[2,3]), y.mean(dim=[2,3])) + std * dist_func(x.std(dim=[2,3]), y.std(dim=[2,3]))
        csim_func = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # Initialize shims (one per diffusion step).
        zts = [torch.zeros_like(empty_embedding, dtype=torch.float32).requires_grad_(False) for _ in range(args.attack_num_inference_steps)]
        
        log.info("Startup timestep {}".format(timesteps[args.start_step].item()))
        # timestep = torch.tensor([140], dtype=torch.long, device=device)
        # noise = torch.randn_like(target_latents)
        # head_start_latents = pipe.scheduler.add_noise(target_latents, noise, timestep)
        head_start_latents = anchor_latents[args.start_step]

        for idx, k in enumerate(args.k):
            zts[k] = zts[k].clone().normal_(0, 0.01)
            for zt in zts:
                zt.requires_grad = False
            zts[k].requires_grad = True
            
            optimizer = torch.optim.Adam([zts[k]], lr=args.lr, weight_decay=1e-3)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            log.info("Optimize latent for step {}, timestep {}".format(k, timesteps[k]))
            for ep in range(args.iters):
                import gc; gc.collect(); torch.cuda.empty_cache()  # Force PyTorch to release cached but unallocated VRAM BEFORE backward pass
                
                optimizer.zero_grad()
                outputs_attack = pipe.generate_with_shims(
                    empty_prompt,
                    num_images_per_prompt=1,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.attack_num_inference_steps,
                    height=args.image_length,
                    width=args.image_length,
                    output_type='latent',
                    grad_step=k,
                    shims=zts,
                    shim_type="text_embeddings",
                    head_start_step=args.start_step,
                    head_start_latents=head_start_latents,
                    shortcut_step=args.shortcut_step
                )
                latents_xtm1 = outputs_attack.inter_latents_next
                
                mu_k = (alphas_cumprod[k]) ** 0.5
                std_k = (1 - alphas_cumprod[k]) ** 0.5

                # Compute losses — all three are active:
                # loss_norm: penalizes shim norm, depends directly on zts[k]
                # loss_align: aligns attacked latents to anchor, backprops through checkpointed UNet -> zts[k]
                # loss_semantic: preserves text semantics, depends directly on zts[k]
                loss_norm = torch.max(torch.zeros(1).to(device), torch.tensor(args.eps[idx]).to(device) - zts[k].norm())
                loss_align = args.gamma1 * dist_func(latents_xtm1, anchor_latents[k-1]).mean()
                loss_semantic = args.gamma2 * (1 - csim_func((empty_embedding + zts[k]).mean(dim=1), empty_embedding.mean(dim=1)).mean())
                loss_image = torch.tensor(0.0, device=device)

                loss = loss_norm + loss_align + loss_semantic
                
                loss.backward()
                log.info("ep {} | loss {:.6f}, loss norm {:.6f}, loss latent {:.6f}, loss semantic {:.6f}, loss image {:.6f}, grad norm {:.6f}".format(
                    ep, loss.item(), loss_norm.item(), loss_align.item(), loss_semantic.item(), loss_image.item(), zts[k].grad.norm()))

                torch.nn.utils.clip_grad_norm_([zts[k]], max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                
                if ep == args.iters - 1:
                    final_attack_images = outputs_attack.images.detach().cpu()
                
                # Delete large vars manually to assist GC
                del outputs_attack, latents_xtm1, loss, loss_align, loss_norm, loss_semantic
                gc.collect(); torch.cuda.empty_cache()
        
        log.info("Shims norm: " + ", ".join("{:.2f}".format(zt.norm().item()) for zt in zts))
        
        # --- Save and Evaluate the Attacked Image ---
        attack_filename = os.path.join(args.output_folder, f"image_attack_{i:04d}_{j:02d}.png")
        with torch.no_grad():
            decoded_attack = pipe.decode_latents(final_attack_images.to(device))
        attack_image_w_pil = tensor_to_pil(decoded_attack)[0]
        attack_image_w_pil.save(attack_filename)
        
        # decoded_wm_attack = wmarker.decode(attack_filename)
        # wm_bit_acc_attack, wm_success_attack = get_bit_acc_success(
        #     decoded_wm_attack,
        #     expected_text=args.watermark_text,
        #     threshold=24/32
        # )
        # log.info("Attacked Image {:04d} - Decoded Watermark: {} | Bit Accuracy: {:.2f}% (Success: {})".format(
        #     i, decoded_wm_attack, wm_bit_acc_attack * 100, wm_success_attack))
        # attack_accuracies.append(wm_bit_acc_attack)
        
        # log.info("Image {:04d} Summary | Original: {:.2f}%, Reversed: {:.2f}%, Attacked: {:.2f}%".format(
        #     i, wm_bit_acc_original*100, wm_bit_acc_reversed*100, wm_bit_acc_attack*100))
        
        # avg_original = sum(original_accuracies) / len(original_accuracies) if original_accuracies else 0
        # avg_reversed = sum(reversed_accuracies) / len(reversed_accuracies) if reversed_accuracies else 0
        # avg_attack = sum(attack_accuracies) / len(attack_accuracies) if attack_accuracies else 0
        # log.info("Average Bit Accuracy - Original: {:.2f}%, Reversed: {:.2f}%, Attacked: {:.2f}%".format(
        #     avg_original*100, avg_reversed*100, avg_attack*100))
