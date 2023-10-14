# Temporal-Image-AnimateDiff

This is a retrain of AnimateDiff to be conditional on an input image. It is trained over the inpainting stable diffusion model to gain more input channels.

To run this, download the motion module from here: https://huggingface.co/CiaraRowles/Temporal-Image and place it in the motion_module folder

Then run the animate command similar to the original AnimateDiff repository, something like:

python -m scripts.animate --config "prompts\\5-RealisticVision.yaml" --image_path "D:\\images\\mona.png"  --W 256 --H 256

and set up the config mostly as before but with these three keys at the start as shown in the provided config files.

base: "models/StableDiffusionInpainting"
The base must be the diffusers inpainting model here: https://huggingface.co/runwayml/stable-diffusion-inpainting

vae: "models/StableDiffusion"
This must be the original 1.5 stable diffusion model, it needs the vae from here due to a diffusers issue: https://huggingface.co/runwayml/stable-diffusion-v1-5

path: "models/DreamBooth_LoRA/realisticVisionV20_v20.inpainting.safetensors"
and this must be your Dreambooth model, it must be an inpainting model, you can convert existing models to inpainting models.

Notes:

This model *only* works well at 256x256 right now, i'll be releasing the 512x512 retrain somewhere, but it's not as good, you would be better off just upscaling in comfyui.

In terms of making it work well, consider the prompts are not you telling the model what you want with this, you're guiding the generation for what is in the input image, if the input image and the prompts do not align, it will not work.
You may have to try a few seeds per generation to get a nice image in addition.

## Acknowledgements
Thanks to the AniamteDiff devs: https://github.com/guoyww/AnimateDiff
