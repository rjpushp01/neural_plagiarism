import argparse
import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def get_watermark_mask(orig_path, target_path):
    orig = cv2.imread(orig_path)
    vis = cv2.imread(target_path)
    
    diff = cv2.absdiff(orig, vis)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Dilate mask to ensure we cover the edges of the watermark
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(thresh, kernel, iterations=2)
    return Image.fromarray(mask)

def main():
    parser = argparse.ArgumentParser(description="Inpainting attack on visible watermarks")
    parser.add_argument('--target_folder', required=True, help='Folder containing target (watermarked) images')
    parser.add_argument('--original_folder', required=True, help='Folder containing original unwatermarked images')
    parser.add_argument('--output_folder', default='./outputs/', help='Folder for saving output images')
    parser.add_argument('--start', type=int, default=0, help='Starting index for processing images')
    parser.add_argument('--end', type=int, default=10, help='Ending index for processing images')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    parser.add_argument('--model_id', default='runwayml/stable-diffusion-inpainting', help='Model ID for the diffusion pipeline')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of diffusion steps for inference')
    parser.add_argument('--image_length', type=int, default=256, help='Length of the image (square assumed)')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.enable_attention_slicing()

    ori_img_paths = glob.glob(os.path.join(args.target_folder, '*.*'))
    ori_img_paths = sorted([path for path in ori_img_paths if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
    ori_img_paths = ori_img_paths[args.start:args.end]

    print(f"Total images: {len(ori_img_paths)}")

    for i, ori_img_path in enumerate(ori_img_paths):
        img_name = os.path.basename(ori_img_path)
        orig_path = os.path.join(args.original_folder, img_name)
        
        if not os.path.exists(orig_path):
            print(f"Original image {orig_path} not found. Skipping.")
            continue

        target_img_pil = Image.open(ori_img_path).convert("RGB")
        target_img_pil = target_img_pil.resize((args.image_length, args.image_length))
        
        # Calculate mask based on original size
        orig_img_pil = Image.open(orig_path).convert("RGB")
        orig_img_pil = orig_img_pil.resize((args.image_length, args.image_length))
        
        # Save temporary files to use cv2 easily
        target_img_pil.save('/tmp/tgt_tmp.png')
        orig_img_pil.save('/tmp/orig_tmp.png')
        
        mask_image = get_watermark_mask('/tmp/orig_tmp.png', '/tmp/tgt_tmp.png')
        
        # Inpaint
        prompt = ""
        generator = torch.Generator(device=device).manual_seed(0)
        
        inpaint_output = pipe(
            prompt=prompt,
            image=target_img_pil,
            mask_image=mask_image,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        ).images[0]
        
        attack_filename = os.path.join(args.output_folder, f"image_attack_{i:04d}_00.png")
        inpaint_output.save(attack_filename)
        print(f"Saved {attack_filename}")

if __name__ == '__main__':
    main()
