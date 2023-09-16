import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple
from collections import OrderedDict

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.data.dataset import WebVid10M
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print



def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank



def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 3,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    print(pretrained_model_path)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae_Dir = "/home/holo/AnimateDiff-512/models/StableDiffusionVae"
    
    vae          = AutoencoderKL.from_pretrained(vae_Dir, subfolder="vae")
    #vae = AutoencoderKL
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet", 
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
    )

        
    missing_keys_initial, _ = unet.load_state_dict(unet.state_dict(), strict=False)
    loaded__checkpoint_params_names = ""
    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        print(f"from checkpoint: {unet_checkpoint_path}")
        checkpoint = torch.load(unet_checkpoint_path, map_location="cpu")
    
        if "global_step" in checkpoint:
            zero_rank_print(f"global_step: {checkpoint['global_step']}")
    
        loaded_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
        # Check for "module." prefix and remove
        new_loaded_state_dict = {key.replace("module.", "") if key.startswith("module.") else key: value 
                                 for key, value in loaded_state_dict.items()}
    
        current_model_dict = unet.state_dict()

        
        loaded__checkpoint_params_names = set(loaded_state_dict.keys())

        
        #new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        #                  for k, v in zip(current_model_dict.keys(), new_loaded_state_dict.values())}
    
        missing_after_load, unexpected = unet.load_state_dict(new_loaded_state_dict, strict=False)
        print(f"missing keys after loading checkpoint: {len(missing_after_load)}, unexpected keys: {len(unexpected)}")
        assert len(unexpected) == 0
    
        if set(missing_after_load).issubset(set(missing_keys_initial)):
            print("All initially missing keys were filled by the checkpoint!")
        else:
            still_missing = set(missing_keys_initial).intersection(set(missing_after_load))
            print(f"Some keys are still missing after loading the checkpoint: {still_missing}")

        # If the checkpoint has optimizer state, load it
        #if "optimizer_state_dict" in checkpoint:
        #    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
        # If the checkpoint has an epoch state, you can also retrieve it
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    #tokenizer.requires_grad_(False)
    # Set unet trainable parameters
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

                
        
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    trainable_params_names = [(name, param) for name, param in unet.named_parameters() if param.requires_grad]
    trainable_params_set = set([name for name, _ in trainable_params_names])

    #print(f"Trainable params names: {[name for name, _ in trainable_params_names]}")
    in_checkpoint_not_trainable = loaded__checkpoint_params_names.difference(trainable_params_set)
    print(f"Parameters in loaded checkpoint but not in trainable params: {in_checkpoint_not_trainable}")
    #
    # Determine which parameters are in the trainable_params but not in the loaded checkpoint
    in_trainable_not_checkpoint = trainable_params_set.difference(loaded__checkpoint_params_names)
    print(f"Parameters in trainable params but not in loaded checkpoint: {in_trainable_not_checkpoint}")
    all_params_in_unet = set([name for name, _ in unet.named_parameters()])
    
    # Check if the parameters from the checkpoint are in unet
    in_checkpoint_but_not_in_unet = in_checkpoint_not_trainable.difference(all_params_in_unet)
    
    #if not in_checkpoint_but_not_in_unet:
    #    print("All parameters from the checkpoint are present in the unet model.")
    #else:
    #    print(f"Parameters present in checkpoint but not in the unet model: {all_params_in_unet}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    train_dataset = WebVid10M(**train_data, is_image=image_finetune)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")
                    
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]

            # Generate a random mask: mask the same channels across an image, but a random proportion of images in the video image group
            mask_proportion = torch.rand(1).item()
            mask_flag = torch.rand(pixel_values.shape[0]) < mask_proportion

            # Create a single channel mask for the images
            single_channel_mask = torch.zeros(pixel_values.shape[0], 1, *pixel_values.shape[2:])

            for i in range(pixel_values.shape[0]):
                if mask_flag[i]:
                    single_channel_mask[i, ...] = 1

            # Expand the single channel mask to match the channel dimensions of pixel_values
            image_mask = single_channel_mask.expand_as(pixel_values).to(pixel_values.device)

            # Apply the mask to the pixel_values to black out the masked regions
            masked_pixel_values = pixel_values * (1 - image_mask)

            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    
                    # Encode the masked_pixel_values to get masked_latents
                    masked_pixel_values = rearrange(masked_pixel_values, "b f c h w -> (b f) c h w")
                    masked_latents = vae.encode(masked_pixel_values).latent_dist
                    masked_latents = masked_latents.sample()
                    masked_latents = rearrange(masked_latents, "(b f) c h w -> b c f h w", f=video_length)

                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample().cuda()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                    # For image_finetune, we'll also need to generate masked_latents
                    masked_latents = vae.encode(masked_pixel_values).latent_dist
                    masked_latents = masked_latents.sample().cuda()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Generate a random mask: mask the same channels across an image, but a random proportion of images in the video image group
            
            
            if single_channel_mask.dim() == 5:  # Confirming it has 5 dimensions
                single_channel_mask = single_channel_mask.squeeze(-1)  # Squeezing the last dimension
            #print(f"single channels shape {single_channel_mask.shape}")
            latents_shape = latents.shape[-3:]
            if single_channel_mask.dim() == 4:  # 4D, so might need to add a depth/length dimension
                single_channel_mask = single_channel_mask.unsqueeze(2)  # Assuming the depth/length is the missing dimension
            elif single_channel_mask.dim() > 5:
                raise ValueError("Unexpected number of dimensions in single_channel_mask")
            
            #print(f"single channels shape {single_channel_mask.shape}")
            mask = F.interpolate(single_channel_mask, size=latents_shape, mode='nearest')
            #print(f"mask after interp {mask.shape}")
            mask = mask.cuda()

            # For the inpainting pipeline, we'll use the masked_latents as input
            #latent_model_input = torch.cat([latents] * 2)  # Assuming do_classifier_free_guidance is True for simplicity
            #latent_model_input = self.scheduler.scale_model_input(latent_model_input, timesteps)
            #if num_channels_unet == 9:
            latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)


            # Predict the noise residual using the inpainting model
            #print(f"noise shape {noisy_latents.shape} mask shape {mask.shape} masked_latents {masked_latents.shape}")
            #print(f"latent shape {latent_model_input.shape}")
            #print(f"text embeddings train  shape {encoder_hidden_states.shape}")
            

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Implement Gradient Accumulation
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if mixed_precision_training:
                    """ >>> gradient clipping >>> """
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    """ <<< gradient clipping <<< """
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    """ >>> gradient clipping >>> """
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    """ <<< gradient clipping <<< """
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0):
                save_path = os.path.join(output_dir, f"checkpoints")
                save_path = os.path.join(save_path, f"checkpoint-epoch-{global_step}.ckpt")
                save_checkpoint(unet,optimizer,trainable_params,save_path)

                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                pixel_values = batch["pixel_values"].to(local_rank)
                video_length = pixel_values.shape[1]
                print(f"pixel values shape {pixel_values.shape}")
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)

                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

                #mask_values = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], device=latents.device)

                # Randomly decide which frames to mask out completely.
                num_frames = pixel_values.shape[1]
                mask_frames = torch.randint(0, 2, (num_frames,)).bool()

                # Create a mask tensor based on mask_frames.
                masks = torch.zeros_like(pixel_values[:, :, 0:1, :, :])
                for i, frame_mask in enumerate(mask_frames):
                    if frame_mask:
                        masks[:, i, :, :, :] = 1
                print(masks.shape)
                inverted_mask = 1 - masks

                # Convert videos to latent space
               
                with torch.no_grad():
                    if not image_finetune:

                        masked_pixel_values = pixel_values * inverted_mask
                        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                        masked_pixel_values = rearrange(masked_pixel_values, "b f c h w -> (b f) c h w")
                        #masks = masks.unsqueeze(0)
                        # Generate masked_latents
                        print(f"pixel values {pixel_values.shape} mask {inverted_mask.shape}")
                        
                        masked_latents = vae.encode(masked_pixel_values).latent_dist
                        masked_latents = masked_latents.sample()
                        masked_latents = rearrange(masked_latents, "(b f) c h w -> b c f h w", f=video_length)
                        
                        latents = vae.encode(pixel_values).latent_dist
                        latents = latents.sample()
                        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    else:
                        latents = vae.encode(pixel_values).latent_dist
                        latents = latents.sample()
                        
                        # For image_finetune, we'll also need to generate masked_latents
                        masked_pixel_values = pixel_values * inverted_mask
                        masked_latents = vae.encode(masked_pixel_values).latent_dist
                        masked_latents = masked_latents.sample()

                    latents = latents * 0.18215

                sample = validation_pipeline(
                    prompt       = batch["text"],
                    generator    = generator,
                    video_length = train_data.sample_n_frames,
                    height       = height,
                    width        = width,
                    latents      = latents,
                    masks        = masks,
                    masked_latents = masked_latents,  # Pass the masked_latents to the validation pipeline
                    **validation_data,
                ).videos

                print("saving sample")
                save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                samples.append(sample)
            
                if not image_finetune:
                    samples = torch.concat(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                    save_videos_grid(samples, save_path)
                    
                    
                    
                else:
                    samples = torch.stack(samples)
                    save_path = f"{output_dir}/samples/sample-{global_step}.png"
                    torchvision.utils.save_image(samples, save_path, nrow=4)
    
                logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()

    
def save_checkpoint(unet, optimizer, trainable_params, mm_path, epoch=None, global_step=None):
    # Convert the list of trainable parameters to a set for faster lookup
    trainable_params_set = set(id(p) for p in trainable_params)
    
    # Extract the current state of the UNet model
    state_dict = unet.state_dict()
    
    checkpoint = {
        'state_dict': state_dict, 
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step
    }
    
    torch.save(checkpoint, mm_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
