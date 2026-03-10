import os
import glob
import subprocess
import json
import numpy as np
from PIL import Image
import sys

# Add parent directory to path to import io_utils and watermarker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watermarker import InvisibleWatermarker
from optim_utils import bytearray_to_bits

def calculate_psnr(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert("RGB"), dtype=np.float32)
    img2 = np.array(Image.open(img2_path).convert("RGB"), dtype=np.float32)
    
    # Resize img2 to img1 size if they differ
    if img1.shape != img2.shape:
        img2 = np.array(Image.open(img2_path).convert("RGB").resize((img1.shape[1], img1.shape[0])), dtype=np.float32)
        
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)

def get_bit_acc(decode_text, expected_text='test'):
    try:
        expected_bits = bytearray_to_bits(expected_text.encode('utf-8'))
        if isinstance(decode_text, bytes):
            wm_bits = bytearray_to_bits(decode_text)
        elif isinstance(decode_text, str):
            wm_bits = bytearray_to_bits(decode_text.encode('utf-8'))
        else:
            return 0.0
        
        # Make sure arrays are the same length
        min_len = min(len(expected_bits), len(wm_bits))
        if min_len == 0:
            return 0.0
        bit_acc = (np.array(expected_bits[:min_len]) == np.array(wm_bits[:min_len])).mean()
        return float(bit_acc)
    except Exception as e:
        print(f"Error calculating bit accuracy: {e}")
        return 0.0

def run_attack(target_folder, output_folder, start_step, k_list, eps, iters, num_images, mask_attack=False, gamma3=1e-3, original_folder=None):
    print(f"Running attack on {target_folder}...")
    k_str = ' '.join(map(str, k_list))
    # Note: run_attack.py takes --k as a list
    command = [
        sys.executable, 'run_attack.py',
        '--target_folder', target_folder,
        '--start', '0',
        '--end', str(num_images),
        '--gpu', '0',
        '--gpu', '0',
        '--start_step', str(start_step),
        '--iters', str(iters),
        '--output_folder', output_folder,
        '--image_length', '256',  # 4x less VRAM: attention maps scale as O(resolution^2)
    ]
    # --k takes multiple arguments
    command.extend(['--k'] + [str(x) for x in k_list])
    # --eps must match the length of k_list
    command.extend(['--eps'] + [str(eps) for _ in k_list])
    
    command.extend(['--gamma3', str(gamma3)])
    if mask_attack:
        command.append('--mask_attack')
    if original_folder is not None:
        command.extend(['--original_folder', original_folder])
    
    # Set CUDA allocator config to reduce fragmentation between sequential subprocess calls
    import copy
    env = copy.copy(os.environ)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'
    
    print(f"Executing: {' '.join(command)}")
    
    # Use Popen to stream output line-by-line so we don't look stuck
    process = subprocess.Popen(
        command, 
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        env=env
    )
    
    for line in process.stdout:
        print(line, end='', flush=True)
        
    process.wait()
    
    if process.returncode != 0:
        print(f"Error running attack (Return code: {process.returncode}). Check logs above for details.")
    else:
        print("Attack sub-process finished successfully.")
        
    return sorted(glob.glob(os.path.join(output_folder, 'image_attack_*_00.png')))

def run_inpaint_attack(target_folder, original_folder, output_folder, num_images):
    print(f"Running Inpainting attack on {target_folder}...")
    command = [
        sys.executable, 'run_inpaint.py',
        '--target_folder', target_folder,
        '--original_folder', original_folder,
        '--end', str(num_images),
        '--gpu', '0',
        '--output_folder', output_folder,
        '--image_length', '256',
    ]
    
    import copy
    env = copy.copy(os.environ)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'
    
    print(f"Executing: {' '.join(command)}")
    process = subprocess.Popen(
        command, 
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        env=env
    )
    
    for line in process.stdout:
        print(line, end='', flush=True)
        
    process.wait()
    return sorted(glob.glob(os.path.join(output_folder, 'image_attack_*_00.png')))

def evaluate_pipeline():
    base_dir = './test_images'
    original_dir = os.path.join(base_dir, 'original')
    visible_dir = os.path.join(base_dir, 'visible')
    invisible_dir = os.path.join(base_dir, 'invisible')
    
    results = {
        'visible_baseline': [],
        'visible_masked': [],
        'visible_gamma3': [],
        'visible_inpaint': [],
        'invisible': []
    }
    
    original_images = sorted(glob.glob(os.path.join(original_dir, '*.jpg')))
    
    # 1. Evaluate Visible Watermarks
    print("\n--- Evaluating Visible Watermarks ---")
    
    # Run Baseline (for reference)
    vis_out_dir = './evaluation_outputs/visible_baseline'
    os.makedirs(vis_out_dir, exist_ok=True)
    print("Stage 1a: Running EVASION BASELINE on visible watermarks...")
    run_attack(visible_dir, vis_out_dir, start_step=15, k_list=[25, 45], eps=10, iters=5, num_images=10)
    
    # Run Masked Attack
    vis_masked_dir = './evaluation_outputs/visible_masked'
    os.makedirs(vis_masked_dir, exist_ok=True)
    print("Stage 1b: Running MASKED ATTACK on visible watermarks...")
    run_attack(visible_dir, vis_masked_dir, start_step=15, k_list=[25, 45], eps=10, iters=5, num_images=10, mask_attack=True, original_folder=original_dir)

    # Run Gamma3 Parameter Attack
    vis_gamma_dir = './evaluation_outputs/visible_gamma3'
    os.makedirs(vis_gamma_dir, exist_ok=True)
    print("Stage 1c: Running IMAGE LOSS ATTACK on visible watermarks...")
    run_attack(visible_dir, vis_gamma_dir, start_step=15, k_list=[25, 45], eps=10, iters=5, num_images=10, gamma3=1000.0, original_folder=original_dir)
    
    # Run Inpainting Attack
    vis_inpaint_dir = './evaluation_outputs/visible_inpaint'
    os.makedirs(vis_inpaint_dir, exist_ok=True)
    print("Stage 1d: Running INPAINTING ATTACK on visible watermarks...")
    run_inpaint_attack(visible_dir, original_dir, vis_inpaint_dir, num_images=10)

    print("Stage 2: Calculating PSNR metrics for visible watermarks...")
    
    dirs_to_evaluate = [
        ('visible_baseline', vis_out_dir),
        ('visible_masked', vis_masked_dir),
        ('visible_gamma3', vis_gamma_dir),
        ('visible_inpaint', vis_inpaint_dir),
    ]

    for result_key, out_dir in dirs_to_evaluate:
        attacked_images = sorted(glob.glob(os.path.join(out_dir, 'image_attack_*_00.png')))
        for i, (orig, attacked) in enumerate(zip(original_images, attacked_images)):
            if os.path.exists(attacked):
                psnr = calculate_psnr(orig, attacked)
                results[result_key].append({
                    'image_idx': i,
                    'original': orig,
                    'attacked': attacked,
                    'psnr': psnr
                })
                
    # 2. Evaluate Invisible Watermarks
    print("\n--- Evaluating Invisible Watermarks ---")
    invis_out_dir = './evaluation_outputs/invisible'
    os.makedirs(invis_out_dir, exist_ok=True)
    
    print("Stage 3/4: Running evasion attack on invisible watermarks (this may take a while)...")
    # Parameters for invisible: late start_step, smaller k
    run_attack(invisible_dir, invis_out_dir, start_step=45, k_list=[47], eps=10, iters=5, num_images=10)
    
    print("Stage 4/4: Decoding watermarks and calculating metrics for invisible watermarks...")
    invisible_watermarker = InvisibleWatermarker(wm_text='test', method='dwtDctSvd')
    attacked_invisible = sorted(glob.glob(os.path.join(invis_out_dir, 'image_attack_*_00.png')))
    invis_watermarked_images = sorted(glob.glob(os.path.join(invisible_dir, '*.jpg')))
    
    for i, (orig, wmarked, attacked) in enumerate(zip(original_images, invis_watermarked_images, attacked_invisible)):
        if os.path.exists(attacked):
            psnr = calculate_psnr(orig, attacked)
            
            # Decode watermark from BEFORE attack
            wm_before = invisible_watermarker.decode(wmarked)
            bit_acc_before = get_bit_acc(wm_before, 'test')
            
            # Decode watermark from AFTER attack
            wm_after = invisible_watermarker.decode(attacked)
            bit_acc_after = get_bit_acc(wm_after, 'test')
            
            results['invisible'].append({
                'image_idx': i,
                'original': orig,
                'attacked': attacked,
                'psnr': psnr,
                'bit_acc_before': bit_acc_before,
                'bit_acc_after': bit_acc_after
            })
            
    os.makedirs('./evaluation_outputs', exist_ok=True)
    with open('./evaluation_outputs/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Evaluation pipeline completed. Results saved to ./evaluation_outputs/metrics.json")
    
if __name__ == '__main__':
    evaluate_pipeline()
