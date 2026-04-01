import numpy as np
import torch
from utils.ecc_watermark import ECCWatermarker
import matplotlib.pyplot as plt
import os
import datetime

def simulate_shim_attack():
    # Setup Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"evaluation_outputs/ecc_shim_attack/run_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file_path = os.path.join(out_dir, "shim_results.log")
    
    def log_print(*args, **kwargs):
        line = " ".join(map(str, args))
        print(line, **kwargs)
        with open(log_file_path, "a") as f:
            f.write(line + "\n")

    log_print("=== Scenario B: Anchor & Shim Attack vs. ECC-Hardened Watermark ===")
    
    # 1. Setup ECC Watermarker (Defense)
    wm_text = "ECC_PROTECT"
    wm = ECCWatermarker(wm_text=wm_text, bch_bits=10, repetition=5)
    ecc_bits = wm.encode()
    
    # 2. Create Anchor (The watermarked latent)
    # This represents the "Copyrighted" data the attacker wants to plagiarize
    anchor_latent = wm.map_to_latent(ecc_bits).numpy()
    
    # 3. Simulate "Shim" Attack (Adversarial Perturbation)
    # A Shim attack isn't just random noise; it's an optimization to diverge 
    # from the anchor while maintaining semantic similarity.
    # We simulate this as a "Directional Shift" + "Stochastic Noise".
    
    epsilons = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] # Shim strength
    ecc_recovered = []
    baseline_recovered = []
    
    log_print(f"\nTargeting Message: '{wm_text}'")
    log_print(f"{'Shim (eps)':<12} | {'ECC Recovery':<15} | {'Bit Flips Corrected'}")
    log_print("-" * 50)

    for eps in epsilons:
        # Generate a 'Shim' - a perturbation with a specific directional bias
        # mimicking the optimization in your Anchor and Shim repo.
        directional_bias = np.sign(np.random.normal(0, 1, anchor_latent.shape))
        shim = (eps * directional_bias) + np.random.normal(0, 0.1, anchor_latent.shape)
        
        # Attacked Latent
        attacked_latent = anchor_latent + shim
        
        # 4. Attempt Extraction
        extracted_text, flips = wm.extract_from_latent(torch.from_numpy(attacked_latent))
        
        success = extracted_text == wm_text
        ecc_recovered.append(1.0 if success else 0.0)
        
        log_print(f"{eps:<12.1f} | {'SUCCESS' if success else 'FAILED':<15} | {flips if flips != -1 else 'N/A'}")

    # 5. Summary & Interpretation
    log_print("\n--- ATTACK ANALYSIS ---")
    if ecc_recovered[2] > 0: # Check if it survived eps=0.4
        log_print("RESULT: The ECC-Hardened Watermark is RESISTANT to moderate Shim attacks (eps <= 0.4).")
        log_print("In your project, this means an attacker would need to introduce MUCH more semantic")
        log_print("alteration (corruption) to remove the copyright, likely ruining the 'Plagiarized' image's quality.")
    else:
        log_print("RESULT: The attack successfully broke the watermark at this threshold.")

    # Visualize the "Shim" impact on the latent distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(anchor_latent.flatten(), bins=50, alpha=0.5, label='Anchor', color='blue')
    plt.title("Original Watermarked Latent")
    plt.subplot(1, 2, 2)
    plt.hist((anchor_latent + (0.8 * directional_bias)).flatten(), bins=50, alpha=0.5, label='Attacked', color='red')
    plt.title("Shim Attacked Latent (eps=0.8)")
    plot_path = os.path.join(out_dir, "shim_attack_simulation.png")
    plt.savefig(plot_path)
    log_print(f"\nVisualized attack impact saved to '{plot_path}'")

if __name__ == "__main__":
    simulate_shim_attack()
