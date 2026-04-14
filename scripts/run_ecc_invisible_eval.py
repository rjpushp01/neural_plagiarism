import os
import sys
import glob
import subprocess
import json
import time
import numpy as np
import cv2
import torch
from PIL import Image

try:
    from pytorch_msssim import ssim as pt_ssim
    has_ssim = True
except ImportError:
    has_ssim = False
    print("Warning: pytorch_msssim not found. SSIM will be skipped.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ecc_invisible_watermark import ECCInvisibleWatermarker

# For metrics
import lpips
try:
    lpips_model = lpips.LPIPS(net='vgg').cuda()
    has_lpips = True
except Exception as e:
    has_lpips = False
    print(f"LPIPS not available: {e}")

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(img1, img2):
    if not has_ssim: return -1.0
    # convert np images [H,W,C] (0-255) to float tensors [1,C,H,W] (0-1)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return float(pt_ssim(t1, t2, data_range=1.0).item())

def compute_lpips(img1_path, img2_path):
    if not has_lpips: return -1.0
    
    def load_tensor(path):
        img = Image.open(path).convert('RGB').resize((512, 512))
        arr = np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
        
    t1 = load_tensor(img1_path)
    t2 = load_tensor(img2_path)
    with torch.no_grad():
        return float(lpips_model(t1, t2).item())

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
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("[ERROR] Attack subprocess failed!", flush=True)
    else:
        print("[SUCCESS] Attack returned naturally.")
        
    return sorted(glob.glob(os.path.join(output_folder, 'image_attack_*_00.png')))

def main():
    print("\n" + "="*70, flush=True)
    print("ECC Invisible Watermark (Option B) Full Evaluation", flush=True)
    print("="*70 + "\n", flush=True)
    
    output_dir = "evaluation_outputs/ecc_invisible_eval"
    wm_dir = os.path.join(output_dir, "watermarked_images")
    attack_dir = os.path.join(output_dir, "shim_attack")
    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(attack_dir, exist_ok=True)
    
    orig_dir = "test_images/original"
    images = sorted(glob.glob(os.path.join(orig_dir, '*.png')) + glob.glob(os.path.join(orig_dir, '*.jpg')))
    images = images[:10]  # Take 10 images
    num_images = len(images)
    
    wm = ECCInvisibleWatermarker(wm_text="test", bch_bits=5, repetition=5)
    
    results = []
    
    print("--- PHASE 1: Embedding ECC Invisible Watermark ---", flush=True)
    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        out_path = os.path.join(wm_dir, f"image_{i:04d}.png")
        
        # We need to resize to exactly 512x512 before embedding so the algorithm works identically
        img_pil = Image.open(img_path).convert('RGB').resize((512, 512))
        img_pil.save(out_path) # save temp
        
        # Embed
        wm.encode(out_path, out_path)
        
        # Extract pre-attack (Self check)
        ext = wm.decode(out_path)
        psnr_val = compute_psnr(np.array(img_pil), np.array(Image.open(out_path)))
        print(f"Image {i}: PSNR vs original: {psnr_val:.2f}dB | Pre-attack BER: {ext['ber_final']:.2%} | Rec: {ext['message_recovered']}", flush=True)
        
        results.append({
            'idx': i,
            'name': fname,
            'orig_path': img_path,
            'wm_path': out_path,
            'psnr_wm': psnr_val,
            'pre_attack': ext
        })
        
    print("\n--- PHASE 2: Shim Attack ---", flush=True)
    attacked_files = run_attack_subprocess(wm_dir, attack_dir, num_images)
    
    print("\n--- PHASE 3: Metrics & Evaluation ---", flush=True)
    for i, attack_file in enumerate(attacked_files):
        if i >= len(results): break
        
        res = results[i]
        
        # Metrics
        img_orig = np.array(Image.open(res['orig_path']).convert('RGB').resize((512, 512)))
        img_wm = np.array(Image.open(res['wm_path']).convert('RGB'))
        img_atk = np.array(Image.open(attack_file).convert('RGB'))
        
        psnr_vs_orig = compute_psnr(img_orig, img_atk)
        ssim_val = compute_ssim(img_orig, img_atk)
        lpips_val = compute_lpips(res['orig_path'], attack_file)
        
        # Extract post-attack
        ext = wm.decode(attack_file)
        
        print(f"\n--- Image {i} ---")
        print(f"  Attack Quality: PSNR(orig)={psnr_vs_orig:.2f}dB | SSIM={ssim_val:.4f} | LPIPS={lpips_val:.4f}")
        print(f"  Post-Attack BER: raw={ext['ber_raw']:.2%} -> voted={ext['ber_voted']:.2%} -> final={ext['ber_final']:.2%}")
        print(f"  Recovered: {ext['message_recovered']} | BCH fixes: {ext['bch_corrections']}")
        
        res['atk_path'] = attack_file
        res['attack_quality'] = {
            'psnr': psnr_vs_orig,
            'ssim': ssim_val,
            'lpips': lpips_val
        }
        res['post_attack'] = ext
        
    # Save JSON summary
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nEvaluation Complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
