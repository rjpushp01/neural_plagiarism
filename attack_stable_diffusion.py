import copy
from functools import partial
from typing import Callable, List, Optional, Union, Any, Dict, Tuple

import numpy as np
import PIL
from PIL import Image

import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, get_cosine_schedule_with_warmup
import torch.nn.functional as F
import math

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
from diffusers.utils import logging, BaseOutput

from modified_stable_diffusion import ModifiedStableDiffusionPipeline
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from contextlib import nullcontext
import pdb

class AttackStableDiffusionPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]
    latents: Optional[torch.FloatTensor]
    inter_latents: Optional[torch.FloatTensor]
    inter_latents_next: Optional[torch.FloatTensor]
    
class AttackStableDiffusionPipeline(InversableStableDiffusionPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super(AttackStableDiffusionPipeline, self).__init__(vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                requires_safety_checker)
    
    @torch.enable_grad()
    def get_image_latents_with_grad(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    @torch.enable_grad()
    def decode_latents_with_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        return image

    def decoder_inv(self, x):
        """
        decoder_inv calculates latents z of the image x by solving optimization problem ||E(x)-z||,
        not by directly encoding with VAE encoder. "Decoder inversion"

        INPUT
        x : image data (1, 3, 512, 512)
        OUTPUT
        z : modified latent data (1, 4, 64, 64)

        Goal : minimize norm(e(x)-z)
        """
        input = x.clone().float()

        z = self.get_image_latents(x).clone().float()
        z.requires_grad_(True)

        loss_function = torch.nn.MSELoss(reduction='sum')
        
        optimizer = torch.optim.Adam([z], lr=0.1)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

        for i in self.progress_bar(range(100)):
            x_pred = self.decode_image_for_gradient_float(z)

            loss = loss_function(x_pred, input)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
        return z

    def decode_image_for_gradient_float(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = (1 / 0.18215 * latents).to(dtype=self.vae.dtype)
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0).float()
        return image

    def _add_shim(
            self, 
            latents: torch.FloatTensor, 
            text_embeddings: torch.FloatTensor, 
            shim: torch.FloatTensor, 
            shim_type: str
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""
        Apply shim perturbations to the given latents or text embeddings.
        """
        if shim_type == "latents":
            return latents + shim.to(latents.dtype), text_embeddings
        elif shim_type == "text_embeddings":
            return latents, text_embeddings + shim.to(text_embeddings.dtype)
        elif shim_type == "latents_fft":
            latents_fft = torch.fft.fft2(latents)
            noisy_latents_fft = latents_fft + shim
            latents = torch.fft.ifft2(noisy_latents_fft).real
            return latents, text_embeddings
        elif shim_type == "both":
            return latents+shim[0].to(latents.dtype), text_embeddings+ shim[1].to(text_embeddings.dtype)
        elif shim_type == "both_fft":
            latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
            noisy_latents_fft = latents_fft + shim[0].to(latents_fft.dtype)
            latents = torch.fft.ifft2(torch.fft.ifftshift(noisy_latents_fft, dim=(-1, -2))).real
            return latents, text_embeddings+ shim[1].to(text_embeddings.dtype)
        else:
            raise Exception("Unsupported shim type.")

    def _reverse_next(self, latents, t, text_embeddings, do_classifier_free_guidance, guidance_scale, shortcut, extra_step_kwargs):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        if torch.is_grad_enabled():
            # Use gradient checkpointing to recompute UNet activations on-the-fly during backward
            # instead of storing them. This reduces peak VRAM from ~3GB to ~500MB.
            def unet_forward(lmi, te):
                return self.unet(lmi, t, encoder_hidden_states=te).sample
            noise_pred = torch.utils.checkpoint.checkpoint(
                unet_forward, latent_model_input, text_embeddings, use_reentrant=False
            )
        else:
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        if shortcut:
            # Compute shortcut latent. Modify from https://github.com/ruocwang/dpo-diffusion/blob/main/src/model/sd_pipeline.py line 199-218
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)

            sqrt_alpha_prod = alphas_cumprod[t] ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(latents.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(latents.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
            latents = (latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
        else:
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        return latents

    # override the call function to output intermediate latents.
    # Remove the watermarking parameters as the it were not known for the attacker.
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        head_start_latents: Optional[Union[torch.FloatTensor, list]] = None,
        head_start_step: Optional[int] = None,
        shortcut_step : Optional[int] = -1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        if head_start_latents is not None:
            latents = head_start_latents
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )

        init_latents = copy.deepcopy(latents)
        inter_latents = None
        inter_latents_next = None
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
         # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if head_start_step and i < head_start_step:
                continue
            
            shortcut = (i == shortcut_step)
            latents = self._reverse_next(latents, t, text_embeddings, do_classifier_free_guidance, guidance_scale, shortcut, extra_step_kwargs)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                # progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        # 8. Post-processing
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        else:
            image = self.decode_latents(latents)
            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
            # 10. Convert to PIL
            if output_type == "pil":
                image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return AttackStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, init_latents=init_latents, latents=latents, inter_latents=inter_latents, inter_latents_next=inter_latents_next)

    def generate_with_shims(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        grad_step : Optional[int] = -1,
        shortcut_step : Optional[int] = -1,
        shims: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        shim_type: Optional[str] = "text_embeddings",
        head_start_latents: Optional[Union[torch.FloatTensor, list]] = None,
        head_start_step: Optional[int] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        if head_start_latents is not None:
            latents = head_start_latents
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
        
        init_latents = latents
        inter_latents = None
        inter_latents_next = None

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 7. Denoising loop
        if shims is None or len(shims) != len(timesteps):
            raise Exception("Incorrect shims")
        
        # Generated image can be opimized if there exists a shortcut. Optional but not necssary.
        enable_image_grad = False

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(timesteps):
            if head_start_step is not None and i < head_start_step:
                continue
                
            shortcut = (i == shortcut_step)
            enable_grad = (i == grad_step)
            
            ctx = torch.enable_grad() if enable_grad else torch.no_grad()

            with ctx:
                latents, text_embeddings = self._add_shim(latents, text_embeddings, shims[i], shim_type)
                
                # snapshot the inter latents at timestep grad_step
                inter_latents = latents.clone() if enable_grad else inter_latents
                
                # _reverse_next uses gradient checkpointing internally when grad is enabled,
                # so only ~500MB peak activation memory (not 3GB) for the backward pass
                latents = self._reverse_next(latents, t, text_embeddings, do_classifier_free_guidance, guidance_scale, shortcut, extra_step_kwargs)

                # snapshot the inter_latents_next latents at timestep grad_step
                inter_latents_next = latents.clone() if enable_grad else inter_latents_next

                # Jump out of the loop.
                if shortcut: 
                    enable_image_grad = enable_grad
                    break

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        # 8. Post-processing
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        else:
            image = self.decode_latents_with_grad(latents) if enable_image_grad else self.decode_latents(latents)
            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
            # 10. Convert to PIL
            if output_type == "pil":
                image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return AttackStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, 
            init_latents=init_latents, latents=latents, inter_latents=inter_latents, inter_latents_next=inter_latents_next)