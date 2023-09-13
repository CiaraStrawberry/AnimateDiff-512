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

import torchvision.io as io

def load_video_to_tensor(video_path, num_frames_to_extract=16, step=2):
    """
    Load a video and convert it to tensor format by extracting a certain number of frames.
    
    Args:
        video_path (str): Path to the video file.
        num_frames_to_extract (int): The number of frames to extract from the video.
        step (int): Extract every 'step' frame. Default is 1 (i.e., every frame).

    Returns:
        torch.Tensor: Tensor with dimensions (batch_size, frames, channels, height, width).
    """
    metadata = io.read_video_metadata(video_path)
    total_frames = metadata['video_duration'] * metadata['video_fps']
    
    # Compute which frames to start and end on
    frames_to_skip = (total_frames - num_frames_to_extract * step) // (2 * step)
    start_frame = int(frames_to_skip * step)
    end_frame = int(start_frame + num_frames_to_extract * step)
    
    video_tensor, _, _ = io.read_video(video_path, start_pts=start_frame/metadata['video_fps'], end_pts=end_frame/metadata['video_fps'], pts_unit='sec')  # Returns video (T, H, W, C)
    
    # Subsample video by the step
    video_tensor = video_tensor[::step]

    # Convert video tensor from (T, H, W, C) to (C, T, H, W)
    video_tensor = video_tensor.permute(3, 0, 1, 2)
    
    # Add batch dimension (assuming the video is just one sample)
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor




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
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
            
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
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

            pipeline.to("cuda")
            ### <<< create validation pipeline <<< ###

            generator = torch.Generator(device='cuda')  # Adjust the device as necessary
            global_seed = config[config_key].random_seed[0]  # Assuming this works with your configuration
            generator.manual_seed(global_seed)
            
            height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
            width = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size
            
            samples = []
            sample_idx = 0  # initialize sample index 
            
            prompts = model_config.prompt
            n_prompts = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            input_video_tensor = load_video_to_tensor(args.video_path)  
            video_length = input_video_tensor.shape[2]

            with torch.no_grad():
                input_video_tensor = rearrange(input_video_tensor, "b c f h w -> (b f) c h w")
                latents = vae.encode(input_video_tensor).latent_dist
                latents = latents.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)


            
            # Convert the input video tensor to latents
            with torch.no_grad():
                latents = vae.encode(input_video_tensor).latent_dist
                latents = latents.sample()
            
            for prompt_idx, (prompt, n_prompt) in enumerate(zip(prompts, n_prompts)):
            
                random_seed = model_config.get("seed", [-1])[prompt_idx]
                if random_seed != -1: 
                    generator.manual_seed(random_seed)
                else:
                    generator.seed()
                print(f"current seed: {torch.initial_seed()}")
            
                # Using the new pipeline
                sample = validation_pipeline(
                    prompt       = prompt,  # Assuming the prompt is just a text string
                    generator    = generator,
                    video_length = train_data.sample_n_frames,
                    height       = height,
                    width        = width,
                    latents      = latents,
                    masks        = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], device='cuda'),  # Adjust the device
                    **validation_data,
                ).videos
            
                # Saving the Sample
                prompt_cleaned = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt_cleaned}.gif")
                print(f"save to {savedir}/sample/{prompt_cleaned}.gif")
            
                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--video_path"),           type=str, default="videos/input.mp4"
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)
