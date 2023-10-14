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


def load_image_to_tensor(image_path, target_size=(256, 256), save_dir='./saved_images'):
    """
    Load an image, crop it to match the target aspect ratio, save it, and convert it to tensor format.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): The target height and width for the image. Default is (256, 256).
        save_dir (str): Directory to save the cropped images before converting to tensor.

    Returns:
        torch.Tensor: Tensor with dimensions (channels, height, width).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load the image at {image_path}")

    h, w, _ = image.shape
    target_ratio = target_size[1] / target_size[0]
    image_ratio = w / h

    if image_ratio > target_ratio:
        # Crop width to match the target ratio
        new_width = int(h * target_ratio)
        start_w = (w - new_width) // 2
        image = image[:, start_w:start_w+new_width]
    elif image_ratio < target_ratio:
        # Crop height to match the target ratio
        new_height = int(w / target_ratio)
        start_h = (h - new_height) // 2
        image = image[start_h:start_h+new_height, :]

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
            tokenizer    = CLIPTokenizer.from_pretrained(model_config.base, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(model_config.base, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(model_config.vae, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(model_config.base, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

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
                    input_image_tensor = input_image_tensor.repeat(video_length, 1, 1, 1).cuda()  # Repeat the image tensor for all frames, also move it to GPU
                    masks = torch.ones(1,video_length, 1, args.H, args.W, device='cuda')  # Expanded the mask shape to match the latent's
                    masks[:, 0, 0] = 0
                    print(f"input image tensor shape: {input_image_tensor.shape}")
                    with torch.no_grad():
                        pixel_values = rearrange(input_image_tensor, "f c h w -> (f) c h w")
                        latents = vae.encode(pixel_values).latent_dist
                        latents = latents.sample()
                        latents = rearrange(latents, "(f) c h w -> c f h w", f=video_length)
                    
                        # Generate the masked pixel values and latents
                        
                        first_frame = input_image_tensor.unsqueeze(0)
                        print(f"first frame shape {first_frame.shape}")
                        masked_latents = vae.encode(rearrange(first_frame, "b f c h w -> (b f) c h w")).latent_dist
                        masked_latents = masked_latents.sample()
                        masked_latents = rearrange(masked_latents, "(b f) c h w -> b c f h w", f=video_length)
                       # masked_latents = masked_latents.unsqueeze(0)
                    for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                        generator.manual_seed(random_seed)    
                        config[config_key].random_seed.append(random_seed)
                        print(f"latents shape: {latents.shape}")
                        
                        # Using the pipeline
                        sample = pipeline(
                            generator    = generator,
                            prompt=prompt,
                            width=args.W,
                            height=args.H,
                            video_length=args.L,
                            latents=None,
                            masks=masks,
                            masked_latents=masked_latents   # Passing the masked latents
                        ).videos

                        # Saving the Sample
                        prompt_cleaned = "-".join((prompt.replace("/", "").split(" ")[:10]))
                        save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt_cleaned}.gif")
                        print(f"Saved to {savedir}/sample/{prompt_cleaned}.gif")

                        sample_idx += 1

    #samples = torch.concat(samples)
    #save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    #OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--image_path",            type=str, default="videos/input.mp4")
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--H", type=int, default=256)

    args = parser.parse_args()
    main(args)
