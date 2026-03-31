import os
import torch
import numpy as np
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from utils.ecc_watermark import ECCWatermarker
import argparse

def main(args):
    device = "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # 1. Load Pipeline
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    ).to(device)
    
    # 2. Setup ECC Watermarker
    # Standard latent shape for SD v1.5 is (4, 64, 64) for 512x512 images
    wm = ECCWatermarker(wm_text=args.text, bch_bits=args.bch_bits, repetition=args.repetition)
    bits = wm.encode()
    
    # 3. Create Watermarked Initial Latent (z_T)
    # The paper embeds the watermark in the INITIAL latent z_T
    z_T = wm.map_to_latent(bits).to(device).to(torch.float32)
    
    # 4. Generate Image from the Watermarked Latent
    print(f"Generating watermarked image for prompt: '{args.prompt}'")
    generator = torch.Generator(device).manual_seed(args.seed)
    
    # We use the pipe to generate an image starting from our custom z_T
    # In Stable Diffusion, we can pass latents to the pipeline
    image = pipe(
        prompt=args.prompt,
        latents=z_T, # Inject the watermarked noise
        num_inference_steps=20, # Reduced for CPU speed
        guidance_scale=7.5,
    ).images[0]
    
    # 5. Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "ecc_watermarked.png")
    image.save(out_path)
    print(f"Saved watermarked image to {out_path}")
    
    # 6. Verification (Immediate Extraction from generated image)
    # To extract, we must first INVERT the image back to z_T
    print("Verifying extraction via inversion...")
    
    # Convert PIL to tensor
    img_tensor = (np.array(image.resize((512, 512))) / 127.5 - 1.0)
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device).to(torch.float32)
    
    # Get image latents
    with torch.no_grad():
        img_latents = pipe.get_image_latents(img_tensor, sample=False)
        
        # Invert to find z_T
        inverted_latents = pipe.forward_diffusion(
            latents=img_latents,
            text_embeddings=pipe.get_text_embedding(""), # Unconditional inversion
            num_inference_steps=20, # Match generation steps
        )
        
    extracted_text, flips = wm.extract_from_latent(inverted_latents)
    print(f"Extracted Text: {extracted_text}")
    print(f"BCH Corrections: {flips}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A professional portrait of Elon Musk")
    parser.add_argument("--text", type=str, default="COPYRIGHT_2026")
    parser.add_argument("--output_dir", type=str, default="./ecc_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bch_bits", type=int, default=8)
    parser.add_argument("--repetition", type=int, default=5)
    args = parser.parse_args()
    main(args)
