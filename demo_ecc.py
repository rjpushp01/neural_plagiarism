import os
import sys
import glob
import subprocess
import json
import time
import gc
import random
import numpy as np
import torch
import cv2
from PIL import Image
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, DDIMScheduler

from attack_stable_diffusion import AttackStableDiffusionPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ecc_invisible_watermark import ECCInvisibleWatermarker
from utils.ecc_watermark import ECCWatermarker
from utils.ldpc_watermark import LDPCWatermarker

# Metrics
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

def run_attack_subprocess(target_folder, output_folder):
    cmd = [
        sys.executable, "run_attack.py",
        "--target_folder", target_folder,
        "--output_folder", output_folder,
        "--start", "0", "--end", "1",
        "--start_step", "45",
        "--k", "47",
        "--eps", "10.0",
        "--iters", "5",
        "--image_length", "512",
        "--gpu", "0"
    ]
    print(f"\nRunning attack command: {' '.join(cmd)}", flush=True)
    
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    subprocess.run(cmd, env=env)
    attacks = sorted(glob.glob(os.path.join(output_folder, 'image_attack_*_00.png')))
    if not attacks:
        raise RuntimeError("Attack failed to produce an output image.")
    return attacks[0]

def embed_ddim_shallow(img_path, out_path, wm_class, shallow_step=15):
    pipe = load_pipeline()
    empty_prompt = ""
    do_cfg = False
    precomputed_emb = pipe._encode_prompt(empty_prompt, "cuda:0", 1, do_cfg, None).detach().clone()
    
    img_pil = Image.open(img_path).convert('RGB').resize((512, 512))
    img_tensor = (np.array(img_pil).astype(np.float32) / 127.5) - 1.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to("cuda:0").half()
    
    z_0 = get_vae_latent(pipe, img_tensor)
    
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
            
            latents = backward_ddim(latents, alpha_prod_t_prev, alpha_prod_t, noise_pred)
            
    z_15 = latents.clone()
    z_15_w = wm_class.embed_into_latent(z_15, margin=0.5).to("cuda:0")
    
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
    
    del pipe, precomputed_emb, img_tensor, z_0, latents, z_15, z_15_w, latents_rec, gen_tensor
    if 'noise_pred' in locals():
        del noise_pred
    if 'latent_model_input' in locals():
        del latent_model_input
    gc.collect()
    torch.cuda.empty_cache()

def extract_ddim_shallow(img_path, wm_class, shallow_step=15):
    pipe = load_pipeline()
    empty_prompt = ""
    do_cfg = False
    precomputed_emb = pipe._encode_prompt(empty_prompt, "cuda:0", 1, do_cfg, None).detach().clone()
    
    img_pil = Image.open(img_path).convert('RGB').resize((512, 512))
    atk_tensor = (np.array(img_pil).astype(np.float32) / 127.5) - 1.0
    atk_tensor = torch.from_numpy(atk_tensor).permute(2, 0, 1).unsqueeze(0).to("cuda:0").half()
    
    z_0_atk = get_vae_latent(pipe, atk_tensor)
    
    ddim_scheduler = DDIMScheduler.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="scheduler")
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
            
    ext = wm_class.extract_detailed(latents_atk)
    del pipe, precomputed_emb, atk_tensor, z_0_atk, latents_atk
    if 'noise_pred' in locals():
        del noise_pred
    if 'latent_model_input' in locals():
        del latent_model_input
    gc.collect()
    torch.cuda.empty_cache()
    return ext

def main():
    print("="*80)
    print("🚀 Neural Plagiarism: End-to-End Comprehensive ECC Pipeline Demo")
    print("Runs 3 completely different architectures against a random test image.")
    print("="*80)

    # Output Structure
    base_dir = "demo_outputs/ecc"
    experiments = ["VAE_BCH", "DDIM_Shallow_BCH", "DDIM_Shallow_LDPC"]
    for exp in experiments:
        os.makedirs(f"{base_dir}/{exp}/watermarked", exist_ok=True)
        os.makedirs(f"{base_dir}/{exp}/attacked", exist_ok=True)

    # 1. Random Image Selection
    orig_dir = "test_images/original"
    images = sorted(glob.glob(os.path.join(orig_dir, '*.png')) + glob.glob(os.path.join(orig_dir, '*.jpg')))
    if not images:
        print("No images found in test_images/original.")
        return
        
    rand_img_path = random.choice(images)
    print(f"\n📸 Randomly Selected Image: {os.path.basename(rand_img_path)}")
    
    # Save a perfect 512x512 copy of original
    orig_512_path = os.path.join(base_dir, "original_512x512.png")
    orig_pil = Image.open(rand_img_path).convert('RGB').resize((512, 512))
    orig_pil.save(orig_512_path)
    
    # Init Models & Metrics
    # Init Watermarkers
    wm_vae_bch = ECCWatermarker(wm_text="test", bch_bits=5, repetition=5, target_channels='dual')
    wm_shallow_bch = ECCWatermarker(wm_text="test", bch_bits=5, repetition=5, target_channels='dual')
    wm_shallow_ldpc = LDPCWatermarker(wm_text="test", snr=10.0, d_v=3, d_c=6, repetition=5, target_channels='dual')

    all_metrics = {}

    for exp_idx, exp_name in enumerate(experiments, 1):
        print(f"\n" + "-"*60)
        print(f"[{exp_idx}/3] Starting Pipeline: {exp_name}")
        print("-"*60)
        
        wm_dir = f"{base_dir}/{exp_name}/watermarked"
        atk_dir = f"{base_dir}/{exp_name}/attacked"
        
        wm_img_name = f"image_0000.png"
        wm_img_path = os.path.join(wm_dir, wm_img_name)
        
        # --- A. Embedding ---
        print("[1/4] Embedding watermark...")
        if exp_name == "VAE_BCH":
            pipe = load_pipeline()
            img_pil = Image.open(orig_512_path).convert('RGB')
            img_tensor = (np.array(img_pil).astype(np.float32) / 127.5) - 1.0
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to("cuda:0").half()
            
            z_0 = get_vae_latent(pipe, img_tensor)
            z_0_w = wm_vae_bch.embed_into_latent(z_0).to("cuda:0")
            
            with torch.no_grad():
                gen_tensor = pipe.vae.decode(z_0_w / 0.18215).sample
            
            gen_img_np = (gen_tensor / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            gen_img_np = (gen_img_np * 255).astype(np.uint8)
            gen_img = Image.fromarray(gen_img_np)
            gen_img.save(wm_img_path)
            
            del pipe, img_tensor, z_0, z_0_w, gen_tensor
            gc.collect()
            torch.cuda.empty_cache()
            
        elif exp_name == "DDIM_Shallow_BCH":
            embed_ddim_shallow(orig_512_path, wm_img_path, wm_shallow_bch)
        elif exp_name == "DDIM_Shallow_LDPC":
            embed_ddim_shallow(orig_512_path, wm_img_path, wm_shallow_ldpc)
            
        # Calculate Original vs WM stats
        img_orig_np = np.array(Image.open(orig_512_path))
        img_wm_np = np.array(Image.open(wm_img_path))
        psnr_wm = compute_psnr(img_orig_np, img_wm_np)
        print(f"      => Visual Quality PSNR: {psnr_wm:.2f} dB")
        
        # --- B. Attack ---
        print("[2/4] Executing adversarial Anchor & Shim Attack... (this will take a moment)")
        gc.collect()
        torch.cuda.empty_cache()
        
        atk_img_path = run_attack_subprocess(wm_dir, atk_dir)
        
        # Calculate Original vs Attack stats
        img_atk_np = np.array(Image.open(atk_img_path))
        psnr_atk = compute_psnr(img_orig_np, img_atk_np)
        ssim_atk = compute_ssim(img_orig_np, img_atk_np)
        
        lpips_atk = -1.0
        if has_lpips:
            lpips_model = lpips.LPIPS(net='vgg').cuda()
            lpips_atk = compute_lpips(orig_512_path, atk_img_path, lpips_model=lpips_model)
            del lpips_model
            gc.collect()
            torch.cuda.empty_cache()
        
        print(f"      => Attack Damage PSNR: {psnr_atk:.2f} dB")
        print(f"      => Attack SSIM: {ssim_atk:.4f}")
        print(f"      => Attack LPIPS: {lpips_atk:.4f}")
        
        # --- C. Extraction ---
        print("[3/4] Extracting watermark computationally...")
        if exp_name == "VAE_BCH":
            pipe = load_pipeline()
            atk_pil = Image.open(atk_img_path).convert('RGB')
            atk_tensor = (np.array(atk_pil).astype(np.float32) / 127.5) - 1.0
            atk_tensor = torch.from_numpy(atk_tensor).permute(2, 0, 1).unsqueeze(0).to("cuda:0").half()
            
            z_0_atk = get_vae_latent(pipe, atk_tensor)
            ext = wm_vae_bch.extract_detailed(z_0_atk)
            
            del pipe, atk_tensor, z_0_atk
            gc.collect()
            torch.cuda.empty_cache()
            
        elif exp_name == "DDIM_Shallow_BCH":
            ext = extract_ddim_shallow(atk_img_path, wm_shallow_bch)
        elif exp_name == "DDIM_Shallow_LDPC":
            ext = extract_ddim_shallow(atk_img_path, wm_shallow_ldpc)
            
        print(f"      => Post-Attack BER Funnel:")
        print(f"         Raw Damage: {ext.get('ber_raw', 0):.2%}")
        print(f"         Layer 2 Voting: {ext.get('ber_voted', 0):.2%}")
        print(f"         Layer 1 Math Correction: {ext.get('ber_final', 0):.2%}")
        recovered = ext.get('message_recovered', False)
        print(f"      => Extracted Payload Recovery: {'✅ SUCCESS' if recovered else '❌ FAILED'}")
        
        all_metrics[exp_name] = {
            "PSNR_orig_vs_wm": psnr_wm,
            "Attack_Quality": {
                "PSNR": psnr_atk,
                "SSIM": ssim_atk,
                "LPIPS": lpips_atk
            },
            "Extraction": ext
        }

    # Save all metrics
    with open(f"{base_dir}/demo_metrics_summary.json", 'w') as f:
        # Convert any float32 to float for json serialization
        import copy
        def sanitize_dict(d):
            new_d = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    new_d[k] = sanitize_dict(v)
                elif isinstance(v, list):
                    new_d[k] = [sanitize_dict(x) if isinstance(x, dict) else x for x in v]
                elif isinstance(v, np.ndarray):
                    new_d[k] = v.tolist()
                elif isinstance(v, (np.floating, float)):
                    new_d[k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    new_d[k] = int(v)
                elif isinstance(v, (np.bool_, bool)):
                    new_d[k] = bool(v)
                else:
                    new_d[k] = v
            return new_d
        
        json.dump(sanitize_dict(all_metrics), f, indent=4)

    # --- Print Summary Presentation ---
    print("\n" + "="*95)
    print("🏁 FINAL THREE-WAY ARCHITECTURE BENCHMARK DEMO 🏁")
    print(f"Image Used: {os.path.basename(rand_img_path)}")
    print("Saved paths: demo_outputs/ecc/*")
    print("="*95)
    
    print(f"{'Method':<20} | {'WM PSNR':<8} | {'Atk PSNR':<8} | {'Raw BER':<8} | {'Voted BER':<9} | {'Final BER':<9} | {'Recovered':<9}")
    print("-" * 95)
    for exp_name, m in all_metrics.items():
        psnr_wm = m['PSNR_orig_vs_wm']
        psnr_atk = m['Attack_Quality']['PSNR']
        ber_r = m['Extraction'].get('ber_raw', 1.0)
        ber_v = m['Extraction'].get('ber_voted', 1.0)
        ber_f = m['Extraction'].get('ber_final', 1.0)
        rec = "Yes" if m['Extraction'].get('message_recovered', False) else "No"
        
        print(f"{exp_name:<20} | {psnr_wm:>5.2f} dB | {psnr_atk:>5.2f} dB | {ber_r:>7.2%} | {ber_v:>8.2%} | {ber_f:>8.2%} | {rec}")
    print("="*95)


if __name__ == "__main__":
    main()
