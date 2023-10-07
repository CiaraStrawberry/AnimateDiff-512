import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available
import torchvision.io as io

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import cv2


def load_image_to_tensor(image_path, target_size=(512, 512), save_dir='./saved_images', crop_size=None):
    """
    Load an image, crop it, save it, and convert it to tensor format.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): The target height and width for the image. Default is (512, 512).
        save_dir (str): Directory to save the images before converting to tensor.
        crop_size (tuple): The dimensions for cropping the image. If None, no cropping is done.

    Returns:
        torch.Tensor: Tensor with dimensions (channels, height, width).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load the image at {image_path}")

    if crop_size is not None:
        h, w, _ = image.shape
        center_h, center_w = h // 2, w // 2
        crop_h, crop_w = crop_size
        start_h, end_h = center_h - crop_h // 2, center_h + crop_h // 2
        start_w, end_w = center_w - crop_w // 2, center_w + crop_w // 2
        image = image[start_h:end_h, start_w:end_w]

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)

    # Save the image
    save_path = os.path.join(save_dir, f'{os.path.basename(image_path)}')
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f'Image saved at {save_path}')

    tensor_image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    tensor_image = (tensor_image * 2) - 1  # Scale and shift to [-1, 1]
    return tensor_image.unsqueeze(0)  # Add batch dimension




def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []
    
    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

            ### >>> create validation pipeline >>> ###
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_vae_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            # 1. unet ckpt
            # 1.1 motion module

            
            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)
                    
                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                            
                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    

                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)

                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)

                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                       pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

                    motion_module_state_dict = torch.load(motion_module, map_location="cpu")
                    motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
                    new_loaded_state_dict = {key.replace("module.", "") if key.startswith("module.") else key: value 
                                             for key, value in motion_module_state_dict.items()}
                    motion_module_state_dict = new_loaded_state_dict

                    # Element-wise addition of the weights from converted_unet_checkpoint to the motion_module_state_dict
                    for key in motion_module_state_dict:
                        if key in converted_unet_checkpoint:
                            motion_module_state_dict[key] += converted_unet_checkpoint[key]

                    if "global_step" in motion_module_state_dict: 
                        func_args.update({"global_step": motion_module_state_dict["global_step"]})
                    missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)

                    pipeline.to("cuda")
                    ### <<< create validation pipeline <<< ###

                    generator = torch.Generator(device='cuda')  # Adjust the device as necessary
                    global_seed = 5
                    generator.manual_seed(global_seed)

                    samples = []
                    sample_idx = 0  # initialize sample index 

                    prompts = model_config.prompt
                    n_prompts = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt

                    random_seeds = model_config.get("seed", [-1])
                    random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
                    random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
                    config[config_key].random_seed = []

                    input_image_tensor = load_image_to_tensor(args.image_path, target_size=(args.W, args.H))
                    video_length = args.L

                    for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                        current_input_image = input_image_tensor.clone().unsqueeze(0)  # Add a batch dimension: shape becomes [1, f, c, h, w]
                        all_frames = []

                        config[config_key].random_seed.append(random_seed)
                        
                        # Move repeat operation and mask generation out of the inner loop
                        current_input_image = current_input_image.repeat(1, video_length, 1, 1, 1).cuda()
                        masks = torch.ones(1, video_length, 1, args.H, args.W, device='cuda')

                        for iteration in range(args.R):  # x is the number of times you want to loop
                            if iteration != 0:

                                last_frame = all_frames[-1][:, :, -1, :, :]  # Shape: [b, c, h, w]
                                
                                last_frame = 2 * last_frame - 1
                                
                                # Assign it to the first frame of current_input_image
                                current_input_image = last_frame.unsqueeze(1).repeat(1, video_length, 1, 1, 1).cuda()

                            with torch.no_grad():
                                pixel_values = rearrange(current_input_image, "b f c h w -> (b f) c h w")
                                latents_dist = vae.encode(pixel_values).latent_dist
                                
                                # For masked_latents
                                first_frame = current_input_image[:, 0].squeeze(1)  # Shape: [b, c, h, w]
                                masked_latents_dist = vae.encode(first_frame).latent_dist
                                masked_latents_sample = masked_latents_dist.sample()
                                masked_latents = masked_latents_sample.repeat(1, video_length, 1, 1)
                                masked_latents = rearrange(masked_latents, "b (f c) h w -> b c f h w", f=video_length)


                            
                            sample = pipeline(
                                generator=generator,
                                prompt=prompt,
                                width=args.W,
                                height=args.H,
                                video_length=args.L,
                                latents=None,
                                masks=masks,
                                masked_latents=masked_latents
                            ).videos

                            # Append the generated frames to the all_frames list
                            all_frames.append(sample)

                        # Combine video segments along the frame dimension
                        combined_video_tensor = torch.cat(all_frames, dim=2)  # Concatenate along the 'f' dimension

                        prompt_cleaned = "-".join((prompt.replace("/", "").split(" ")[:10]))
                        save_videos_grid(combined_video_tensor, f"{savedir}/sample/{sample_idx}-{prompt_cleaned}.gif")
                        print(f"Saved to {savedir}/sample/{sample_idx}-{prompt_cleaned}.gif")


                        sample_idx += 1

    #samples = torch.concat(samples)
    #save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    #OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-vae",)
    parser.add_argument("--pretrained_vae_path",   type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--image_path",            type=str, default="videos/input.mp4")
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--R", type=int, default=3)
    args = parser.parse_args()
    main(args)
