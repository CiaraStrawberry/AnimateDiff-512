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
from diffusers.utils import WEIGHTS_NAME
import copy
from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import cv2


def load_video_to_tensor(video_path, num_frames_to_extract=16, step=2, target_size=(512, 512)):
    """
    Load a video and convert it to tensor format by extracting a certain number of frames.
    
    Args:
        video_path (str): Path to the video file.
        num_frames_to_extract (int): The number of frames to extract from the video.
        step (int): Extract every 'step' frame. Default is 1 (i.e., every frame).
        target_size (tuple): The target height and width for each frame. Default is (512, 512).

    Returns:
        torch.Tensor: Tensor with dimensions (batch_size, frames, channels, height, width).
    """
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Compute which frames to start and end on
    frames_to_skip = (total_frames - num_frames_to_extract * step) // (2 * step)
    start_frame = 0
    frame_idx = start_frame
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if not cap.isOpened():
        raise ValueError(f"Could not open the video at {video_path}")
    
    while len(frames) < num_frames_to_extract and frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Set the frame position explicitly
        ret, frame = cap.read()
        print(f"Trying to read frame {frame_idx}. Success: {ret}")
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)  # Resize the frame to the target size
            tensor_frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            #alpha_channel = torch.ones_like(tensor_frame[0, :, :])
            #tensor_frame = torch.cat([tensor_frame, alpha_channel.unsqueeze(0)], dim=0)
            
            frames.append(tensor_frame)
        frame_idx += step
    
    cap.release()
    print(f"Total frames: {total_frames}, Start frame: {start_frame}, End frame: frame_idx")
    
    video_tensor = torch.stack(frames)
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
    
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
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            ### >>> create validation pipeline >>> ###
            tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path, subfolder="vae").to(device)            
            unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).to(device)
            initial_unet_state_dict = copy.deepcopy(unet.state_dict())
            
            
            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")
            pipeline.enable_vae_slicing()
            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")

            state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict

            # Remove "module." prefix from keys
            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            unet.load_state_dict(cleaned_state_dict, strict=False)
            # 2. Compare this saved state dict with the state dict of unet after loading the motion module.
            changed_weights = {}
            current_unet_state_dict = unet.state_dict()

            for key in initial_unet_state_dict:
                if torch.any(initial_unet_state_dict[key] != current_unet_state_dict[key]):
                    changed_weights[key] = current_unet_state_dict[key]

            # 3. Create a new state dict with only the differing values.
            changed_state_dict = changed_weights

            # 4. Save this new state dict as a checkpoint.
            torch.save(changed_state_dict, "D:\\temprojects\\sample\\stripped\\changed_weights.ckpt")

           
            #assert len(unexpected) == 0
            
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
                  #         
                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    
                    vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    #text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    #import pdb
                    #pdb.set_trace()
                    #if is_lora:
                    #    pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

                    pipeline.to("cuda")
                    ### <<< create validation pipeline <<< ###

                    generator = torch.Generator(device='cuda')  # Adjust the device as necessary
                    global_seed = 42
                    generator.manual_seed(global_seed)

                    samples = []
                    sample_idx = 0  # initialize sample index 

                    prompts      = model_config.prompt
                    n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt

                    random_seeds = model_config.get("seed", [-1])
                    random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
                    random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
                    config[config_key].random_seed = []

                    input_video_tensor = load_video_to_tensor(args.video_path,target_size=(args.W,args.H))  
                    video_length = 16
                    print(f"inputvideo tensor shape  {input_video_tensor.shape}")

                    with torch.no_grad():
                        pixel_values = rearrange(input_video_tensor, "b f c h w -> (b f) c h w").cuda()

                        #latents = vae.encode(pixel_values).latent_dist

                        #latents = latents.sample()
                        #latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

                    # Generate the mask of same shape as input tensor but with single channel
                    num_frames = input_video_tensor.shape[1]
                    mask_frames = torch.randint(0, 2, (num_frames,)).bool()  # This will give a list of True and False for frames
                    print(f"mask frames {mask_frames}")
                    # Create a mask tensor based on mask_frames.
                    mask = torch.zeros_like(input_video_tensor[:, :, 0:1, :, :])
                    for i, frame_mask in enumerate(mask_frames):
                        if frame_mask:
                            mask[:, i, :, :, :] = 1

                    inverted_mask = 1 - mask

                    # Create masked_latents
                    inverted_mask = inverted_mask.cuda()
                    masked_pixel_values = pixel_values * inverted_mask
                    with torch.no_grad():
                        masked_pixel_values_squeezed = masked_pixel_values.squeeze(0)
                        masked_latents = vae.encode(masked_pixel_values_squeezed).latent_dist
                        masked_latents = masked_latents.sample()
                        masked_latents = rearrange(masked_latents, "(b f) c h w -> b c f h w", f=video_length)

                    for prompt_idx, (prompt, n_prompt) in enumerate(zip(prompts, n_prompts)):
                        
                        config[config_key].random_seed.append(torch.initial_seed())
                        #print(f"latents shape {latents.shape}")
                        print(f"prompt {prompt}")
                        # Using the new pipeline
                        sample = pipeline(
                            prompt       = prompt,  # Assuming the prompt is just a text string
                            generator    = generator,
                            width               = args.W,
                            height              = args.H,
                            video_length        = args.L,
                            latents      = None,
                            masks        = mask.cuda(),
                            masked_latents = masked_latents,
                        ).videos
            
                        # Saving the Sample
                        prompt_cleaned = "-".join((prompt.replace("/", "").split(" ")[:10]))
                        save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt_cleaned}.gif")
                        print(f"save to {savedir}/sample/{prompt_cleaned}.gif")
                
                
                    sample_idx += 1

    #samples = torch.concat(samples)
    #save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    #OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-vae",)
    parser.add_argument("--pretrained_vae_path",   type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--video_path",            type=str, default="videos/input.mp4")
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--H", type=int, default=256)

    args = parser.parse_args()
    main(args)
