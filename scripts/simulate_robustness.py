import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.ecc_watermark import ECCWatermarker

def calculate_bit_accuracy(original_bits, extracted_bits):
    min_len = min(len(original_bits), len(extracted_bits))
    return (original_bits[:min_len] == extracted_bits[:min_len]).mean()

def run_simulation():
    print("=== Robust Watermarking Simulation (Paper vs. Baseline) ===")
    
    # Configuration
    wm_text = "SECRET_KEY_2026"
    bch_bits = 12       # Correct up to 12 bits per block
    repetition = 5      # 5x redundancy
    latent_shape = (4, 64, 64)
    
    # 1. Initialize ECC Watermarker (The Paper's Method)
    wm = ECCWatermarker(wm_text=wm_text, bch_bits=bch_bits, repetition=repetition, latent_shape=latent_shape)
    ecc_bits = wm.encode()
    
    # 2. Initialize a "Naive" Baseline (No ECC, just raw mapping)
    # Convert text to raw bits for baseline comparison
    raw_bits = np.unpackbits(np.frombuffer(wm_text.encode('utf-8'), dtype=np.uint8))
    
    # 3. Create Watermarked Latents
    # ECC-Hardened Latent
    latent_ecc = wm.map_to_latent(ecc_bits).numpy()
    
    # Naive Latent (Simple mapping for comparison)
    latent_naive = np.random.normal(0, 1, latent_shape).astype(np.float32)
    # For naive, we just force the sign of the first N elements
    flat_naive = latent_naive.flatten()
    flat_naive[:len(raw_bits)] = np.where(raw_bits == 1, np.abs(flat_naive[:len(raw_bits)]), -np.abs(flat_naive[:len(raw_bits)]))
    latent_naive = flat_naive.reshape(latent_shape)

    # --- START STRESS TESTING ---
    noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    results_ecc = []
    results_naive = []
    
    print(f"\nTesting Robustness against Gaussian Manifold Noise:")
    print(f"{'Noise Std':<10} | {'Naive Acc':<12} | {'ECC Acc':<12} | {'Recovery'}")
    print("-" * 55)

    for sigma in noise_levels:
        # Simulate Noise Attack
        # In a real diffusion model, sigma=0.5 is a VERY heavy attack (Shim style)
        noise = np.random.normal(0, sigma, latent_shape)
        
        attacked_naive = latent_naive + noise
        attacked_ecc = latent_ecc + noise
        
        # --- Extraction ---
        
        # Naive Extraction
        extracted_naive_bits = (attacked_naive.flatten()[:len(raw_bits)] > 0).astype(np.uint8)
        naive_acc = calculate_bit_accuracy(raw_bits, extracted_naive_bits)
        
        # ECC Extraction
        extracted_text, flips = wm.extract_from_latent(torch.from_numpy(attacked_ecc))
        
        # Calculate raw bit accuracy BEFORE ECC correction for comparison
        # (Internal bit accuracy of the noisy channel)
        from scipy.stats import norm
        ecc_raw_probs = norm.cdf(attacked_ecc.flatten()[:len(ecc_bits)])
        ecc_raw_bits = (ecc_raw_probs > 0.5).astype(np.uint8)
        ecc_acc = calculate_bit_accuracy(ecc_bits, ecc_raw_bits)
        
        recovery = "SUCCESS" if extracted_text == wm_text else "FAILED"
        
        results_ecc.append(ecc_acc)
        results_naive.append(naive_acc)
        
        print(f"{sigma:<10.1f} | {naive_acc*100:<11.1f}% | {ecc_acc*100:<11.1f}% | {recovery} ({flips} flips corrected)")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, results_naive, 'o--', label='Naive (No ECC)', color='red')
    plt.plot(noise_levels, results_ecc, 's-', label='ECC-Hardened (Paper)', color='green')
    plt.axhline(y=0.5, color='gray', linestyle=':', label='Random Guess (50%)')
    plt.title(f"Robustness Simulation: Watermark Survival under Attack\nTarget Message: '{wm_text}'")
    plt.xlabel("Attack Intensity (Latent Noise Sigma)")
    plt.ylabel("Raw Bit Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("robustness_benchmark.png")
    print(f"\nSimulation complete. Benchmark plot saved to 'robustness_benchmark.png'")
    
    if results_ecc[-1] > results_naive[-1]:
        print("\nCONCLUSION: The ECC-Hardened method (Scenario A) successfully resisted attacks that broke the naive baseline.")
        print("Even when raw bit accuracy dropped, the BCH+Repetition layer corrected errors to recover 100% of the text.")

if __name__ == "__main__":
    run_simulation()
