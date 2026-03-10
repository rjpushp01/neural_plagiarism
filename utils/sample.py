import argparse
import copy
from tqdm import tqdm
from statistics import mean, stdev

import os
import torch
import imageio
import torch.optim 

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline 
import open_clip
from optim_utils import *
from io_utils import *

import PIL.Image
import PIL.ImageOps
from PIL import Image

import torch
import matplotlib.pyplot as plt
from log import *

from pycocotools.coco import COCO
import requests
import numpy as np

from watermarking_wrappers.stable_signature_wrapper import StableSignatureWrapper

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import pdb

def tensor_to_pil(images):
    # Move post-processing in decode to here.
    images = (images / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()

    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def main(args):
    
    if not os.path.exists(args.image_folder):
        os.mkdir(args.image_folder)

    log = Log(args.image_folder+"/log.txt")
    log.info(str(args))

    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    
    ldm_config = "./models/v2-inference.yaml"
    ldm_ckpt = "./models/v2-1_512-ema-pruned.ckpt"
    sd2_decoder_weights = "./models/sd2_decoder.pth"
    wrapper = StableSignatureWrapper(ldm_config, ldm_ckpt, sd2_decoder_weights, device)

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        )
    pipe = pipe.to(device)
    


    # if args.reference_model is not None:
    #     ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
    #     ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    if 'coco' in args.dataset:
        coco = COCO('/data1/datasets/COCO/annotations/captions_train2017.json')
        image_ids = coco.getImgIds()
        ids = []
        images = []
    else:
        dataset, prompt_key = get_dataset(args)

    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []
    rc_metrics = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        if 'coco' in args.dataset:
            img_id = image_ids[i]

            img_info = coco.loadImgs(img_id)[0]
            image_url = img_info['coco_url']
            image = Image.open(requests.get(image_url, stream=True).raw)
            imageio.imwrite(args.image_folder+ "/orig_img_{:04d}.png".format(i), np.array(image))

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns]
            current_prompt = captions[0]
        else:
            current_prompt = dataset[i][prompt_key]

        log.info("############### prompt: {}".format(current_prompt))
        set_random_seed(seed)

        # Tree ring watermarking. Inject key pattern into latents.
        if args.watermark_method == "tree_ring":
            # Generate image.
            init_latents_no_w = pipe.get_random_latents()
            outputs_no_w = pipe(
                current_prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_no_w,
                output_type='latent',
                )
            orig_image_no_w = outputs_no_w.images[0]
            orig_image_no_w_pil = tensor_to_pil(outputs_no_w.images)[0]
            imageio.imwrite(args.image_folder + "/orig_generated_img_{:04d}.png".format(i), orig_image_no_w_pil)

            if init_latents_no_w is None:
                set_random_seed(seed)
                init_latents_w = pipe.get_random_latents()
            else:
                init_latents_w = copy.deepcopy(init_latents_no_w)

            # Inject watermark.
            watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
            init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)
            outputs_w = pipe(
                current_prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_w,
                output_type='latent',
                )
            
            orig_image_w_pil = tensor_to_pil(outputs_w.images)[0]
            imageio.imwrite(args.image_folder + "/watermark_img_{:04d}.png".format(i), orig_image_w_pil)
        
        # Stable signature watermarking. Load the fine-tune decoder.
        elif args.watermark_method == "stable_signature":
            outputs_w = wrapper(
                current_prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                output_type='pil',)

            orig_image_w_pil = outputs_w.images[0]
            imageio.imwrite(args.image_folder + "/watermark_img_{:04d}.png".format(i), orig_image_w_pil)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)

    parser.add_argument('--image_folder', default='./images/')

    # tree ring parameters.
    parser.add_argument('--watermark_method', default=None, help="tree_ring, stable_signature")
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)
