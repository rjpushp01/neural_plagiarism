"""ECC Robust Watermarking — Full Evaluation Pipeline

Embedding Strategy: VAE-latent (z_0) embedding
  - Original image → VAE encode → z_0 → embed watermark → VAE decode → watermarked image
  - Extraction: image → VAE encode → z_0 → extract bits
  - This avoids DDIM inversion which is lossy for real photos (~18 dB, content hallucination)
  - VAE round-trip gives ~30-40 dB PSNR with no content hallucination

Resolution: 512×512 (sweet spot for RTX 4050 6GB VRAM)
  - Latent: 64×64×4 = 16,384 elements
  - Dual-channel mode (ch2+ch3): 440 bits / 8192 = 5.4% per high-freq channel
  - Single-channel mode (ch2 only): 440 bits / 4096 = 10.7%

Phases:
  1. VAE-level watermark embedding (fast, high PSNR)
  2. Run Shim attack on watermarked images
  3. Compute comprehensive metrics (PSNR, SSIM, MS-SSIM, LPIPS, BER funnel, latent cosine sim)
  4. Generate plots and JSON summary
"""

import os
import sys
import glob
import json
import argparse
import datetime
import subprocess
import copy
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ecc_watermark import ECCWatermarker
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from utils.image_processing import tensor_to_pil
from torchvision import transforms

# Optional metrics
try:
    from pytorch_msssim import ssim as pt_ssim, ms_ssim as pt_ms_ssim
    has_msssim = True
except ImportError:
    has_msssim = False
    print("Warning: pytorch_msssim not found. SSIM/MS-SSIM will be skipped.")

try:
    import lpips
    has_lpips = True
except ImportError:
    has_lpips = False
    print("Warning: lpips not found. LPIPS metric will be skipped.")

# -------------------------------------------------------------------------
# Metrics Utilities
# -------------------------------------------------------------------------

def compute_psnr(img1_np, img2_np):
    mse = np.mean((img1_np.astype(float) - img2_np.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def load_tensor(img_path, size=None):
    """Load image as [0,1] tensor for SSIM/LPIPS."""
    img = Image.open(img_path).convert('RGB')
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)

def compute_ssim_val(img1_path, img2_path, size=None):
    if not has_msssim:
        return -1.0
    img1 = load_tensor(img1_path, size)
    img2 = load_tensor(img2_path, size)
    if img1.shape != img2.shape:
        img2 = torch.nn.functional.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
    return pt_ssim(img1, img2, data_range=1.0).item()

def compute_msssim(img1_path, img2_path, size=None):
    if not has_msssim:
        return -1.0
    img1 = load_tensor(img1_path, size)
    img2 = load_tensor(img2_path, size)
    if img1.shape != img2.shape:
        img2 = torch.nn.functional.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
    # ms_ssim requires minimum image dimension of 161+
    if img1.shape[2] < 161 or img1.shape[3] < 161:
        return -1.0
    return pt_ms_ssim(img1, img2, data_range=1.0).item()

def compute_lpips_val(img1_path, img2_path, lpips_fn, size=None):
    if not has_lpips or lpips_fn is None:
        return -1.0
    img1 = load_tensor(img1_path, size) * 2 - 1  # scale to [-1, 1]
    img2 = load_tensor(img2_path, size) * 2 - 1
    if img1.shape != img2.shape:
        img2 = torch.nn.functional.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    with torch.no_grad():
        return float(lpips_fn(img1, img2).item())

def latent_cosine_similarity(z1, z2):
    return torch.nn.functional.cosine_similarity(
        z1.reshape(1, -1).float(), z2.reshape(1, -1).float()
    ).item()

# -------------------------------------------------------------------------
# Pipeline Helpers
# -------------------------------------------------------------------------

def load_pipeline(model_id, device):
    """Load the InversableStableDiffusionPipeline (needed for VAE and for shim attack extraction)."""
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe

def image_to_tensor(img, device, dtype=torch.float16):
    """Convert PIL Image to model input tensor [-1, 1]."""
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).to(dtype)

def vae_encode(pipe, img_tensor):
    """Encode image tensor to VAE latent z_0 (deterministic mode)."""
    with torch.no_grad():
        z_0 = pipe.get_image_latents(img_tensor, sample=False)
    return z_0

def vae_decode_to_pil(pipe, z_0):
    """Decode VAE latent z_0 to PIL image."""
    with torch.no_grad():
        decoded = pipe.decode_image(z_0)
        # decoded is in [-1, 1], convert to [0, 1] then to PIL
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
        decoded = (decoded[0] * 255).round().astype(np.uint8)
    return Image.fromarray(decoded)

def invert_image(pipe, img_tensor, device, num_steps=50, guidance=7.5):
    """Encode image to latent, then DDIM-invert to get z_T (used for post-attack extraction)."""
    with torch.no_grad():
        text_emb = pipe._encode_prompt("", device, 1, True, None)
        img_latents = pipe.get_image_latents(img_tensor, sample=False)
        z_T = pipe.forward_diffusion(
            latents=img_latents,
            text_embeddings=text_emb,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
        )
    return z_T

# -------------------------------------------------------------------------
# Attack Runner
# -------------------------------------------------------------------------

def run_attack_subprocess(target_folder, output_folder, num_images,
                          image_length=512, start_step=45, k_list=[47],
                          eps=10, iters=5):
    """Run shim attack as subprocess. Returns list of attacked image paths."""
    target_folder = os.path.abspath(target_folder)
    output_folder = os.path.abspath(output_folder)

    command = [
        sys.executable, 'run_attack.py',
        '--target_folder', target_folder,
        '--start', '0',
        '--end', str(num_images),
        '--gpu', '0',
        '--start_step', str(start_step),
        '--iters', str(iters),
        '--output_folder', output_folder,
        '--image_length', str(image_length),
    ]
    command.extend(['--k'] + [str(x) for x in k_list])
    command.extend(['--eps'] + [str(eps) for _ in k_list])

    env = copy.copy(os.environ)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'

    print(f"\n  Command: {' '.join(command)}")
    print(f"  Working dir: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

    process = subprocess.Popen(
        command,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    for line in process.stdout:
        print(f"  [ATTACK] {line}", end='', flush=True)
    process.wait()

    if process.returncode != 0:
        print(f"  WARNING: Attack subprocess exited with code {process.returncode}")

    # Collect attacked image paths
    attacked = sorted(glob.glob(os.path.join(output_folder, 'image_attack_*_00.png')))
    print(f"  Found {len(attacked)} attacked images.")
    return attacked

# -------------------------------------------------------------------------
# Main Evaluation
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ECC Watermark Full Evaluation Pipeline")
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image resolution (default: 512)')
    parser.add_argument('--skip_watermark', action='store_true',
                        help='Skip Phase 1 if watermarked images already exist')
    parser.add_argument('--skip_attack', action='store_true',
                        help='Skip Phase 2 if attacked images already exist')
    parser.add_argument('--channel_mode', type=str, default='dual', choices=['single', 'dual'],
                        help='Watermark channel mode: single (ch2) or dual (ch2+ch3)')
    parser.add_argument('--bch_bits', type=int, default=5,
                        help='BCH error correction capability')
    parser.add_argument('--repetition', type=int, default=5,
                        help='Repetition code factor')
    parser.add_argument('--wm_text', type=str, default='test',
                        help='Watermark text message')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to evaluate')
    parser.add_argument('--attack_iters', type=int, default=5,
                        help='Shim attack optimization iterations')
    parser.add_argument('--attack_start_step', type=int, default=45,
                        help='Shim attack start step')
    parser.add_argument('--attack_k', type=int, nargs='+', default=[47],
                        help='Shim attack timestep indices')
    parser.add_argument('--attack_eps', type=float, default=10.0,
                        help='Shim attack epsilon bound')
    args = parser.parse_args()

    IMAGE_SIZE = args.image_size
    MODEL_ID = 'Manojb/stable-diffusion-2-1-base'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_STEPS = 50
    GUIDANCE = 7.5

    # Directories
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'evaluation_outputs', 'ecc_evaluation'))
    wm_dir = os.path.join(out_dir, "ecc_watermarked")
    attack_dir = os.path.join(out_dir, "shim_attack")
    plots_dir = os.path.join(out_dir, "plots")

    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(attack_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Logging
    log_file_path = os.path.join(out_dir, "ecc_evaluation.log")
    # Truncate old log
    with open(log_file_path, 'w') as f:
        f.write("")

    def log_print(*a, **kw):
        line = " ".join(map(str, a))
        print(line, **kw)
        with open(log_file_path, "a") as f:
            f.write(line + "\n")

    log_print("=" * 70)
    log_print(f"ECC Robust Watermarking — Full Evaluation Pipeline")
    log_print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log_print(f"Resolution: {IMAGE_SIZE}×{IMAGE_SIZE}")
    log_print(f"Channel mode: {args.channel_mode}")
    log_print(f"Model: {MODEL_ID}")
    log_print(f"Device: {DEVICE}")
    log_print("=" * 70)

    # Init Watermarker
    wm = ECCWatermarker(
        wm_text=args.wm_text,
        bch_bits=args.bch_bits,
        repetition=args.repetition,
        target_channels=args.channel_mode,
    )
    lat_size = IMAGE_SIZE // 8
    cap = wm.get_capacity_info((1, 4, lat_size, lat_size))
    log_print(f"\nWatermark Capacity Analysis:")
    log_print(f"  Message: '{args.wm_text}' -> {cap['num_bits']} bits (after BCH+Rep)")
    log_print(f"  Latent: {lat_size}×{lat_size}×4 = {4*lat_size*lat_size} elements")
    log_print(f"  Target channel utilization: {cap['utilization_pct']:.1f}%")
    log_print(f"  Total latent modification: {cap['total_latent_pct']:.1f}%")
    log_print(f"  Fits: {cap['fits']}")

    if not cap['fits']:
        log_print("ERROR: Watermark bits exceed channel capacity! Increase image size or reduce repetition.")
        return

    # Get source images
    test_img_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'test_images', 'original'))
    target_images = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))[:args.num_images]
    num_images = len(target_images)
    log_print(f"\nFound {num_images} source images in {test_img_dir}")

    if num_images == 0:
        log_print("ERROR: No source images found!")
        return

    # Store latents for post-attack comparison
    watermarked_latents = {}

    # =====================================================================
    # PHASE 1: Generate Watermarked Images
    # =====================================================================
    existing_wm = sorted(glob.glob(os.path.join(wm_dir, 'image_*.png')))

    if args.skip_watermark and len(existing_wm) >= num_images:
        log_print(f"\n--- PHASE 1: SKIPPED (found {len(existing_wm)} existing watermarked images) ---")
        log_print("  Note: Latent-level metrics (cosine sim) won't be available without re-encoding.")
    else:
        log_print(f"\n{'='*70}")
        log_print(f"PHASE 1: VAE-Level Watermark Embedding ({IMAGE_SIZE}×{IMAGE_SIZE})")
        log_print(f"{'='*70}")
        log_print(f"  Strategy: Image → VAE encode → embed in z_0 → VAE decode → watermarked image")
        log_print(f"  (No DDIM inversion — avoids content hallucination on real photos)")

        pipe = load_pipeline(MODEL_ID, DEVICE)

        for i, img_path in enumerate(target_images):
            img_name = os.path.basename(img_path)
            out_path = os.path.join(wm_dir, f"image_{i:04d}.png")

            log_print(f"\n[{i+1}/{num_images}] Processing {img_name}...")

            # Load and resize
            img = Image.open(img_path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            img_tensor = image_to_tensor(img, DEVICE)

            # VAE encode → z_0
            z_0 = vae_encode(pipe, img_tensor)

            # Embed watermark into z_0 (VAE latent, not noise latent)
            z_0_w = wm.embed_into_latent(z_0)
            watermarked_latents[i] = z_0_w.clone().cpu()

            # VAE decode → watermarked image
            z_0_w_device = z_0_w.to(DEVICE).to(torch.float16)
            gen_img = vae_decode_to_pil(pipe, z_0_w_device)
            gen_img = gen_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            gen_img.save(out_path)

            # PSNR of watermarked vs original
            psnr_wm = compute_psnr(np.array(img), np.array(gen_img))
            log_print(f"  PSNR (watermarked vs original): {psnr_wm:.2f} dB")

            # Verify extraction (round-trip: watermarked image → VAE encode → extract)
            gen_tensor = image_to_tensor(gen_img, DEVICE)
            z_0_recovery = vae_encode(pipe, gen_tensor)
            ext = wm.extract_detailed(z_0_recovery)
            log_print(f"  Pre-attack BER: raw={ext['ber_raw']:.2%} -> voted={ext['ber_voted']:.2%} -> final={ext['ber_final']:.2%}")
            log_print(f"  Recovered: {ext['message_recovered']} | BCH fixes: {ext['bch_corrections']}")

            torch.cuda.empty_cache()

        # Free pipeline VRAM before attack
        del pipe
        torch.cuda.empty_cache()
        log_print(f"\nPhase 1 complete. Watermarked images saved to {wm_dir}")

    # =====================================================================
    # PHASE 2: Shim Attack
    # =====================================================================
    existing_atk = sorted(glob.glob(os.path.join(attack_dir, 'image_attack_*_00.png')))

    if args.skip_attack and len(existing_atk) >= num_images:
        log_print(f"\n--- PHASE 2: SKIPPED (found {len(existing_atk)} existing attacked images) ---")
        attacked_files = existing_atk
    else:
        log_print(f"\n{'='*70}")
        log_print(f"PHASE 2: Running Shim Attack ({IMAGE_SIZE}×{IMAGE_SIZE})")
        log_print(f"{'='*70}")

        attacked_files = run_attack_subprocess(
            target_folder=wm_dir,
            output_folder=attack_dir,
            num_images=num_images,
            image_length=IMAGE_SIZE,
            start_step=args.attack_start_step,
            k_list=args.attack_k,
            eps=args.attack_eps,
            iters=args.attack_iters,
        )

        if len(attacked_files) == 0:
            log_print("\nWARNING: No attacked images were produced!")
            log_print("This may be due to OOM. Try reducing --image_size or --attack_iters.")

    # =====================================================================
    # PHASE 3: Comprehensive Metrics
    # =====================================================================
    log_print(f"\n{'='*70}")
    log_print(f"PHASE 3: Computing Metrics")
    log_print(f"{'='*70}")

    # Reload pipeline for latent extraction
    pipe = load_pipeline(MODEL_ID, DEVICE)

    # Init LPIPS
    lpips_fn = None
    if has_lpips:
        lpips_fn = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            lpips_fn = lpips_fn.cuda()

    results = []
    wm_psnr_list = []
    atk_psnr_list = []

    for i, orig_path in enumerate(target_images):
        img_name = os.path.basename(orig_path)
        wm_path = os.path.join(wm_dir, f"image_{i:04d}.png")
        attack_path = os.path.join(attack_dir, f"image_attack_{i:04d}_00.png")

        if not os.path.exists(wm_path):
            log_print(f"\n  WARNING: Watermarked image not found for idx {i}, skipping.")
            continue

        orig_img = Image.open(orig_path).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        wm_img = Image.open(wm_path).convert('RGB')
        orig_np = np.array(orig_img)
        wm_np = np.array(wm_img)

        entry = {
            'image_idx': i,
            'image_name': img_name,
            'original_path': orig_path,
            'watermarked_path': wm_path,
        }

        # --- Watermark Quality Metrics ---
        psnr_wm = compute_psnr(orig_np, wm_np)
        ssim_wm = compute_ssim_val(orig_path, wm_path, IMAGE_SIZE)
        msssim_wm = compute_msssim(orig_path, wm_path, IMAGE_SIZE)
        lpips_wm = compute_lpips_val(orig_path, wm_path, lpips_fn, IMAGE_SIZE)
        wm_psnr_list.append(psnr_wm)

        entry['watermark_quality'] = {
            'psnr': float(psnr_wm),
            'ssim': float(ssim_wm),
            'ms_ssim': float(msssim_wm),
            'lpips': float(lpips_wm),
        }

        log_print(f"\n--- Image {i}: {img_name} ---")
        log_print(f"  Watermark Quality: PSNR={psnr_wm:.2f}dB | SSIM={ssim_wm:.4f} | LPIPS={lpips_wm:.4f}")

        # --- Pre-attack BER (re-extract from watermarked image via VAE encode) ---
        wm_tensor = image_to_tensor(wm_img, DEVICE)
        z_0_wm = vae_encode(pipe, wm_tensor)
        ext_pre = wm.extract_detailed(z_0_wm)

        entry['pre_attack_ecc'] = {
            'ber_raw': ext_pre['ber_raw'],
            'ber_voted': ext_pre['ber_voted'],
            'ber_final': ext_pre['ber_final'],
            'recovered': ext_pre['message_recovered'],
            'bch_corrections': ext_pre['bch_corrections'],
            'avg_vote_margin': ext_pre['avg_vote_margin'],
        }

        log_print(f"  Pre-attack BER: raw={ext_pre['ber_raw']:.2%} -> voted={ext_pre['ber_voted']:.2%} -> final={ext_pre['ber_final']:.2%} | Recovered: {ext_pre['message_recovered']}")

        # --- Post-attack metrics (if attacked image exists) ---
        has_attack = os.path.exists(attack_path)
        entry['attacked_path'] = attack_path if has_attack else None

        if has_attack:
            atk_img = Image.open(attack_path).convert('RGB')
            # Resize attacked image to match if needed
            if atk_img.size != (IMAGE_SIZE, IMAGE_SIZE):
                atk_img = atk_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            atk_np = np.array(atk_img)

            # Image quality after attack
            psnr_atk_orig = compute_psnr(orig_np, atk_np)
            psnr_atk_wm = compute_psnr(wm_np, atk_np)
            ssim_atk = compute_ssim_val(orig_path, attack_path, IMAGE_SIZE)
            msssim_atk = compute_msssim(orig_path, attack_path, IMAGE_SIZE)
            lpips_atk = compute_lpips_val(orig_path, attack_path, lpips_fn, IMAGE_SIZE)
            atk_psnr_list.append(psnr_atk_orig)

            entry['attack_quality'] = {
                'psnr_vs_original': float(psnr_atk_orig),
                'psnr_vs_watermarked': float(psnr_atk_wm),
                'ssim': float(ssim_atk),
                'ms_ssim': float(msssim_atk),
                'lpips': float(lpips_atk),
            }

            log_print(f"  Attack Quality: PSNR(vs orig)={psnr_atk_orig:.2f}dB | PSNR(vs wm)={psnr_atk_wm:.2f}dB | SSIM={ssim_atk:.4f} | LPIPS={lpips_atk:.4f}")

            # Post-attack BER (extract via VAE encode)
            atk_tensor = image_to_tensor(atk_img, DEVICE)
            z_0_atk = vae_encode(pipe, atk_tensor)
            ext_post = wm.extract_detailed(z_0_atk)

            # Latent cosine similarity (VAE latent space)
            cos_sim = latent_cosine_similarity(z_0_wm, z_0_atk)

            entry['post_attack_ecc'] = {
                'ber_raw': ext_post['ber_raw'],
                'ber_voted': ext_post['ber_voted'],
                'ber_final': ext_post['ber_final'],
                'recovered': ext_post['message_recovered'],
                'bch_corrections': ext_post['bch_corrections'],
                'avg_vote_margin': ext_post['avg_vote_margin'],
            }
            entry['latent_cosine_sim'] = float(cos_sim)

            log_print(f"  Post-attack BER: raw={ext_post['ber_raw']:.2%} -> voted={ext_post['ber_voted']:.2%} -> final={ext_post['ber_final']:.2%} | Recovered: {ext_post['message_recovered']}")
            log_print(f"  Latent Cosine Similarity: {cos_sim:.4f}")
        else:
            log_print(f"  [No attacked image available]")
            entry['post_attack_ecc'] = None
            entry['attack_quality'] = None
            entry['latent_cosine_sim'] = None

        results.append(entry)
        torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()

    # =====================================================================
    # PHASE 4: Summary & Plots
    # =====================================================================
    log_print(f"\n{'='*70}")
    log_print(f"PHASE 4: Summary & Visualization")
    log_print(f"{'='*70}")

    # Save results JSON
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    log_print(f"\nMetrics saved to {metrics_path}")

    if len(results) == 0:
        log_print("\nNo results to summarize!")
        return

    # --- Summary Statistics ---
    pre_recovered = sum(1 for r in results if r['pre_attack_ecc']['recovered'])
    pre_ber_raw = np.mean([r['pre_attack_ecc']['ber_raw'] for r in results])
    pre_ber_final = np.mean([r['pre_attack_ecc']['ber_final'] for r in results])
    avg_wm_psnr = np.mean(wm_psnr_list) if wm_psnr_list else 0

    log_print(f"\n--- WATERMARK QUALITY ---")
    log_print(f"  Avg PSNR (watermarked vs original): {avg_wm_psnr:.2f} dB")
    log_print(f"  Pre-attack recovery: {pre_recovered}/{len(results)} ({pre_recovered/len(results)*100:.1f}%)")
    log_print(f"  Pre-attack Avg BER: raw={pre_ber_raw:.2%}, final={pre_ber_final:.2%}")

    # Post-attack stats (only for images that have attacked versions)
    post_results = [r for r in results if r['post_attack_ecc'] is not None]
    if len(post_results) > 0:
        post_recovered = sum(1 for r in post_results if r['post_attack_ecc']['recovered'])
        post_ber_raw = np.mean([r['post_attack_ecc']['ber_raw'] for r in post_results])
        post_ber_final = np.mean([r['post_attack_ecc']['ber_final'] for r in post_results])
        avg_atk_psnr = np.mean(atk_psnr_list) if atk_psnr_list else 0
        avg_cos_sim = np.mean([r['latent_cosine_sim'] for r in post_results])

        log_print(f"\n--- ATTACK ROBUSTNESS ---")
        log_print(f"  Post-attack recovery: {post_recovered}/{len(post_results)} ({post_recovered/len(post_results)*100:.1f}%)")
        log_print(f"  Post-attack Avg BER: raw={post_ber_raw:.2%}, final={post_ber_final:.2%}")
        log_print(f"  Attack Avg PSNR (vs original): {avg_atk_psnr:.2f} dB")
        log_print(f"  Avg Latent Cosine Similarity: {avg_cos_sim:.4f}")
    else:
        log_print(f"\n--- ATTACK ROBUSTNESS ---")
        log_print(f"  No attacked images available for analysis.")

    # --- Plots ---
    try:
        _generate_plots(results, post_results, plots_dir, log_print)
    except Exception as e:
        log_print(f"\nWarning: Plot generation failed: {e}")

    log_print(f"\n{'='*70}")
    log_print(f"Evaluation Complete. Output: {out_dir}")
    log_print(f"{'='*70}")


def _generate_plots(results, post_results, plots_dir, log_print):
    """Generate comprehensive evaluation plots."""

    indices = [r['image_idx'] for r in results]

    # ---- Plot 1: BER Correction Funnel (Pre-attack) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_ber = [r['pre_attack_ecc']['ber_raw'] for r in results]
    voted_ber = [r['pre_attack_ecc']['ber_voted'] for r in results]
    final_ber = [r['pre_attack_ecc']['ber_final'] for r in results]

    ax.plot(indices, raw_ber, 'o-', label='Raw (Channel)', color='#e74c3c', linewidth=2)
    ax.plot(indices, voted_ber, 's-', label='After Repetition Vote', color='#f39c12', linewidth=2)
    ax.plot(indices, final_ber, '^-', label='After BCH Decode', color='#2ecc71', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_title('ECC Error Correction Funnel — Pre-Attack', fontsize=14, fontweight='bold')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Bit Error Rate')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'ber_funnel_pre_attack.png'), dpi=150)
    plt.close(fig)
    log_print(f"  Saved: ber_funnel_pre_attack.png")

    # ---- Plot 2: PSNR Comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))
    wm_psnr = [r['watermark_quality']['psnr'] for r in results]
    ax.bar(indices, wm_psnr, color='#3498db', alpha=0.8, label='Watermarked vs Original')

    if post_results:
        atk_indices = [r['image_idx'] for r in post_results]
        atk_psnr = [r['attack_quality']['psnr_vs_original'] for r in post_results]
        ax.bar([x + 0.35 for x in atk_indices], atk_psnr, width=0.35,
               color='#e74c3c', alpha=0.8, label='Attacked vs Original')

    ax.axhline(30, color='green', linestyle='--', alpha=0.5, label='30dB threshold')
    ax.set_title('PSNR Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('PSNR (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'psnr_comparison.png'), dpi=150)
    plt.close(fig)
    log_print(f"  Saved: psnr_comparison.png")

    if not post_results:
        return

    # ---- Plot 3: BER Funnel Post-Attack ----
    fig, ax = plt.subplots(figsize=(10, 5))
    post_indices = [r['image_idx'] for r in post_results]
    post_raw = [r['post_attack_ecc']['ber_raw'] for r in post_results]
    post_voted = [r['post_attack_ecc']['ber_voted'] for r in post_results]
    post_final = [r['post_attack_ecc']['ber_final'] for r in post_results]

    ax.plot(post_indices, post_raw, 'o-', label='Raw (Channel)', color='#e74c3c', linewidth=2)
    ax.plot(post_indices, post_voted, 's-', label='After Repetition Vote', color='#f39c12', linewidth=2)
    ax.plot(post_indices, post_final, '^-', label='After BCH Decode', color='#2ecc71', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_title('ECC Error Correction Funnel — Post Shim Attack', fontsize=14, fontweight='bold')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Bit Error Rate')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'ber_funnel_post_attack.png'), dpi=150)
    plt.close(fig)
    log_print(f"  Saved: ber_funnel_post_attack.png")

    # ---- Plot 4: Pre vs Post Attack BER ----
    # Match indices
    pre_map = {r['image_idx']: r for r in results}
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    pi = post_indices
    pre_final_matched = [pre_map[idx]['pre_attack_ecc']['ber_final'] for idx in pi]
    post_final_list = [r['post_attack_ecc']['ber_final'] for r in post_results]

    x_pos = np.arange(len(pi))
    ax.bar(x_pos - bar_width/2, pre_final_matched, bar_width,
           label='Pre-Attack BER', color='#2ecc71', alpha=0.8)
    ax.bar(x_pos + bar_width/2, post_final_list, bar_width,
           label='Post-Attack BER', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(idx) for idx in pi])
    ax.set_title('BER Before vs After Shim Attack (Final, after BCH)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Bit Error Rate')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'ber_pre_vs_post.png'), dpi=150)
    plt.close(fig)
    log_print(f"  Saved: ber_pre_vs_post.png")

    # ---- Plot 5: Comprehensive Quality Dashboard ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SSIM
    ax = axes[0, 0]
    ssim_vals = [r['attack_quality']['ssim'] for r in post_results if r['attack_quality']['ssim'] > 0]
    if ssim_vals:
        ax.bar(range(len(ssim_vals)), ssim_vals, color='#9b59b6', alpha=0.8)
        ax.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='0.9 threshold')
        ax.set_title('SSIM (Attacked vs Original)')
        ax.set_ylabel('SSIM')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # LPIPS
    ax = axes[0, 1]
    lpips_vals = [r['attack_quality']['lpips'] for r in post_results if r['attack_quality']['lpips'] >= 0]
    if lpips_vals:
        ax.bar(range(len(lpips_vals)), lpips_vals, color='#e67e22', alpha=0.8)
        ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, label='0.1 threshold')
        ax.set_title('LPIPS (Attacked vs Original, lower=better)')
        ax.set_ylabel('LPIPS')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Cosine Similarity
    ax = axes[1, 0]
    cos_vals = [r['latent_cosine_sim'] for r in post_results]
    ax.bar(range(len(cos_vals)), cos_vals, color='#1abc9c', alpha=0.8)
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='0.9 threshold')
    ax.set_title('Latent Cosine Similarity (WM vs Attacked)')
    ax.set_ylabel('Cosine Sim')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Vote Margins
    ax = axes[1, 1]
    pre_margins = [r['pre_attack_ecc']['avg_vote_margin'] for r in results]
    post_margins = [r['post_attack_ecc']['avg_vote_margin'] for r in post_results]
    x_pos = np.arange(len(post_margins))
    pre_m_matched = [pre_map[r['image_idx']]['pre_attack_ecc']['avg_vote_margin'] for r in post_results]
    ax.bar(x_pos - 0.2, pre_m_matched, 0.35, label='Pre-Attack', color='#2ecc71', alpha=0.8)
    ax.bar(x_pos + 0.2, post_margins, 0.35, label='Post-Attack', color='#e74c3c', alpha=0.8)
    ax.set_title('Avg Vote Margin (higher = more confident)')
    ax.set_ylabel('Margin')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Attack Quality Dashboard', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'quality_dashboard.png'), dpi=150)
    plt.close(fig)
    log_print(f"  Saved: quality_dashboard.png")


if __name__ == "__main__":
    main()
