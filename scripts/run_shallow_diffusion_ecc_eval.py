import os
import sys
import glob
import subprocess
import json
import time
import gc
import numpy as np
import cv2
import torch
from PIL import Image
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, DDIMScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack_stable_diffusion import AttackStableDiffusionPipeline
from log import Log
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from utils.ecc_watermark import ECCWatermarker

# For metrics
try:
    from pytorch_msssim import ssim as pt_ssim
    has_ssim = True
except ImportError:
    has_ssim = False
    print("Warning: pytorch_msssim not found. SSIM will be skipped.")
import lpips
try:
    has_lpips = True
except Exception as e:
    has_lpips = False
    print(f"LPIPS not available: {e}")

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(255.0 / np.sqrt(mse)))

def compute_ssim(img1, img2):
    if not has_ssim: return -1.0
    # convert np images [H,W,C] (0-255) to float tensors [1,C,H,W] (0-1)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return float(pt_ssim(t1, t2, data_range=1.0).item())

def compute_lpips(img1_path, img2_path, lpips_model=None):
    if not has_lpips or lpips_model is None: return -1.0
    def load_tensor(path):
        img = Image.open(path).convert('RGB').resize((512, 512))
        arr = np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        return float(lpips_model(load_tensor(img1_path), load_tensor(img2_path)).item())

def load_pipeline():
    model_id = "Manojb/stable-diffusion-2-1-base"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

    pipe = AttackStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        vae=vae,
        torch_dtype=torch.float16,
    ).to("cuda:0")
    pipe.enable_attention_slicing(1)
    return pipe

def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )

def get_vae_latent(pipe, img_tensor):
    with torch.no_grad():
        encoding_dist = pipe.vae.encode(img_tensor).latent_dist
        z_0 = encoding_dist.mode() * 0.18215
    return z_0

def run_attack_subprocess(target_folder, output_folder, num_images):
    # Same standard attack params
    cmd = [
        sys.executable, "run_attack.py",
        "--target_folder", target_folder,
        "--output_folder", output_folder,
        "--start", "0", "--end", str(num_images),
        "--start_step", "45",
        "--k", "47",
        "--eps", "10.0",
        "--iters", "5",
        "--image_length", "512",
        "--gpu", "0"
    ]
    print(f"\nRunning attack command: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd)
    return sorted(glob.glob(os.path.join(output_folder, 'image_attack_*_00.png')))

def main():
    print("\n" + "="*70, flush=True)
    print("Shallow Diffusion ECC Watermark (Option A) Full Evaluation", flush=True)
    print("="*70 + "\n", flush=True)
    
    output_dir = "evaluation_outputs/shallow_diffusion_ecc_eval"
    wm_dir = os.path.join(output_dir, "watermarked_images")
    attack_dir = os.path.join(output_dir, "shim_attack")
    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(attack_dir, exist_ok=True)
    
    orig_dir = "test_images/original"
    images = sorted(glob.glob(os.path.join(orig_dir, '*.png')) + glob.glob(os.path.join(orig_dir, '*.jpg')))[:10]
    num_images = len(images)
    
def make_json_serializable(d):
    if isinstance(d, dict): return {k: make_json_serializable(v) for k, v in d.items()}
    if isinstance(d, list): return [make_json_serializable(v) for v in d]
    if isinstance(d, np.ndarray): return d.tolist()
    if isinstance(d, (np.floating, np.integer, np.bool_)): return d.item()
    return d

def phase1_embed(images, wm_dir, wm, model_id="Manojb/stable-diffusion-2-1-base", shallow_step=15):
    pipe = load_pipeline()
    results = []
    
    empty_prompt = ""
    do_cfg = False
    precomputed_emb = pipe._encode_prompt(empty_prompt, "cuda:0", 1, do_cfg, None).detach().clone()
    
    print("--- PHASE 1: Embedding Shallow Diffusion Watermark (DDIM Image-to-Image) ---", flush=True)
    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        out_path = os.path.join(wm_dir, f"image_{i:04d}.png")
        
        # 1. Forward to VAE Latent (z_0)
        img_pil = Image.open(img_path).convert('RGB').resize((512, 512))
        img_tensor = (np.array(img_pil).astype(np.float32) / 127.5) - 1.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to("cuda:0").half()
        
        z_0 = get_vae_latent(pipe, img_tensor)
        
        # 1.5. DDIM Switch and Partial Inversion
        ddim_scheduler = DDIMScheduler.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="scheduler")
        ddim_scheduler.set_timesteps(50, device="cuda:0")
        
        latents = z_0 * ddim_scheduler.init_noise_sigma
        
        with torch.no_grad():
            for i, t in enumerate(reversed(ddim_scheduler.timesteps)):
                if i >= shallow_step:
                    break
                
                latent_model_input = ddim_scheduler.scale_model_input(latents, t)
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=precomputed_emb).sample
                
                prev_timestep = t - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps
                alpha_prod_t = ddim_scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = ddim_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else ddim_scheduler.final_alpha_cumprod
                
                # Invert logic: swap alphas
                latents = backward_ddim(latents, alpha_prod_t_prev, alpha_prod_t, noise_pred)
                
        z_15 = latents.clone()
        
        # 2. Embed ECC in z_15 using our sign-nudging
        z_15_w = wm.embed_into_latent(z_15, margin=0.5).to("cuda:0")
        
        # 3. Denoise back to image pixel space perfectly using DDIM
        with torch.no_grad():
            latents_rec = z_15_w.clone()
            residual_timesteps = ddim_scheduler.timesteps[-(shallow_step):]
            
            for t in residual_timesteps:
                latent_model_input = ddim_scheduler.scale_model_input(latents_rec, t)
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=precomputed_emb).sample
                latents_rec = ddim_scheduler.step(noise_pred, t, latents_rec).prev_sample
                
            gen_tensor = pipe.vae.decode(latents_rec / 0.18215).sample
            
        gen_img_np = (gen_tensor / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        gen_img_np = (gen_img_np * 255).astype(np.uint8)
        gen_img = Image.fromarray(gen_img_np)
            
        gen_img.save(out_path)
        psnr_val = compute_psnr(np.array(img_pil), np.array(gen_img))
        
        # 4. Immediate extraction pre-attack sanity check
        gen_tensor_eval = (np.array(gen_img).astype(np.float32) / 127.5) - 1.0
        gen_tensor_eval = torch.from_numpy(gen_tensor_eval).permute(2, 0, 1).unsqueeze(0).to("cuda:0").half()
        z_0_recovery = get_vae_latent(pipe, gen_tensor_eval)
        
        # Re-invert to exactly z_15 
        latents_extr = z_0_recovery * ddim_scheduler.init_noise_sigma
        with torch.no_grad():
            for i, t in enumerate(reversed(ddim_scheduler.timesteps)):
                if i >= shallow_step:
                    break
                latent_model_input = ddim_scheduler.scale_model_input(latents_extr, t)
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=precomputed_emb).sample
                prev_timestep = t - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps
                alpha_prod_t = ddim_scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = ddim_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else ddim_scheduler.final_alpha_cumprod
                latents_extr = backward_ddim(latents_extr, alpha_prod_t_prev, alpha_prod_t, noise_pred)
                
        ext = wm.extract_detailed(latents_extr)
        
        print(f"Image {i}: PSNR vs orig: {psnr_val:.2f}dB | Pre-attack BER: {ext['ber_final']:.2%} | Rec: {ext['message_recovered']}", flush=True)
        
        results.append({
            'idx': i,
            'name': fname,
            'orig_path': img_path,
            'wm_path': out_path,
            'psnr_wm': psnr_val,
            'pre_attack': ext
        })
    return results

def main():
    print("\n" + "="*70, flush=True)
    print("Shallow Diffusion ECC Watermark (Option A) Full Evaluation", flush=True)
    print("="*70 + "\n", flush=True)
    
    output_dir = "evaluation_outputs/shallow_diffusion_ecc_eval"
    wm_dir = os.path.join(output_dir, "watermarked_images")
    attack_dir = os.path.join(output_dir, "shim_attack")
    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(attack_dir, exist_ok=True)
    
    orig_dir = "test_images/original"
    images = sorted(glob.glob(os.path.join(orig_dir, '*.png')) + glob.glob(os.path.join(orig_dir, '*.jpg')))[:10]
    num_images = len(images)
    
    wm = ECCWatermarker(wm_text="test", bch_bits=5, repetition=5, target_channels='dual')
    shallow_step = 15
    
    # Run Phase 1 in strict local scope
    results = phase1_embed(images, wm_dir, wm, shallow_step=shallow_step)
    
    # Aggressively Free VRAM before subprocess
    gc.collect()
    torch.cuda.empty_cache()
        
    print("\n--- PHASE 2: Shim Attack ---", flush=True)
    attacked_files = run_attack_subprocess(wm_dir, attack_dir, num_images)
    
    print("\n--- PHASE 3: Metrics & Evaluation ---", flush=True)
    pipe = load_pipeline() # reload for extraction
    empty_prompt = ""
    do_cfg = False
    precomputed_emb = pipe._encode_prompt(empty_prompt, "cuda:0", 1, do_cfg, None).detach().clone()
    
    lpips_model = lpips.LPIPS(net='vgg').cuda() if has_lpips else None
    
    for i, attack_file in enumerate(attacked_files):
        if i >= len(results): break
        res = results[i]
        
        # Metrics
        img_orig = np.array(Image.open(res['orig_path']).convert('RGB').resize((512, 512)))
        img_atk = np.array(Image.open(attack_file).convert('RGB'))
        
        psnr_vs_orig = compute_psnr(img_orig, img_atk)
        ssim_val = compute_ssim(img_orig, img_atk)
        lpips_val = compute_lpips(res['orig_path'], attack_file, lpips_model=lpips_model)
        
        # Extract post-attack
        atk_tensor = (np.array(img_atk).astype(np.float32) / 127.5) - 1.0
        atk_tensor = torch.from_numpy(atk_tensor).permute(2, 0, 1).unsqueeze(0).to("cuda:0").half()
        z_0_atk = get_vae_latent(pipe, atk_tensor)
        
        # Re-invert to exactly z_15 using DDIM 
        ddim_scheduler = DDIMScheduler.from_pretrained(pipe.config._name_or_path, subfolder="scheduler")
        ddim_scheduler.set_timesteps(50, device="cuda:0")
        latents_atk = z_0_atk * ddim_scheduler.init_noise_sigma
        with torch.no_grad():
            for idx_t, t in enumerate(reversed(ddim_scheduler.timesteps)):
                if idx_t >= shallow_step:
                    break
                latent_model_input = ddim_scheduler.scale_model_input(latents_atk, t)
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=precomputed_emb).sample
                prev_timestep = t - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps
                alpha_prod_t = ddim_scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = ddim_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else ddim_scheduler.final_alpha_cumprod
                latents_atk = backward_ddim(latents_atk, alpha_prod_t_prev, alpha_prod_t, noise_pred)
                
        ext = wm.extract_detailed(latents_atk)
        
        print(f"\n--- Image {i} ---")
        print(f"  Attack Quality: PSNR(orig)={psnr_vs_orig:.2f}dB | SSIM={ssim_val:.4f} | LPIPS={lpips_val:.4f}")
        print(f"  Post-Attack BER: raw={ext['ber_raw']:.2%} -> voted={ext['ber_voted']:.2%} -> final={ext['ber_final']:.2%}")
        print(f"  Recovered: {ext['message_recovered']} | BCH fixes: {ext['bch_corrections']}")
        
        res['atk_path'] = attack_file
        res['attack_quality'] = {'psnr': psnr_vs_orig, 'ssim': ssim_val, 'lpips': lpips_val}
        res['post_attack'] = ext
        
    # Save JSON summary safely
    clean_results = make_json_serializable(results)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(clean_results, f, indent=4)
        
    print(f"\nEvaluation Complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
