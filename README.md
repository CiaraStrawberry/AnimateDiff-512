# Temporal-Image-AnimateDiff

Introducing an AnimateDiff retrain over the sd-inpainting model in order to take an input image 

## Examples

| Input | Prompt | Output |
|:---------:|:-------:|:---------:|
| ![stockman](https://github.com/CiaraStrawberry/Temporal-Image-AnimateDiff/assets/13116982/9ac4bce0-fb33-4a7a-8a9b-02507c9aee5a) | Man on a Beach | ![A man on a beach](https://github.com/CiaraStrawberry/Temporal-Image-AnimateDiff/assets/13116982/15d815fe-d152-4414-8d0f-6101ecab3c9c) |
| ![tay](https://github.com/CiaraStrawberry/Temporal-Image-AnimateDiff/assets/13116982/8c474f50-023b-4b76-a14e-1c7acfda8ea1) | Taylor Swift | ![Taylor Swift](https://github.com/CiaraStrawberry/Temporal-Image-AnimateDiff/assets/13116982/ded0683e-c1e1-4330-bb88-93b113da5d04) |
| ![beachw](https://github.com/CiaraStrawberry/Temporal-Image-AnimateDiff/assets/13116982/0842a400-19da-4ef6-86fe-e0cc94236815) | Woman on a Beach | ![1-a-woman](https://github.com/CiaraStrawberry/Temporal-Image-AnimateDiff/assets/13116982/5f273d01-e6b6-430e-b463-0aaf0271da59) |


## Setup & Configuration

1. **Download Motion Module**
   - Motion module from [HuggingFace](https://huggingface.co/CiaraRowles/Temporal-Image)
   - Place it in the `motion_module` folder.

2. **Configuration Keys**:
   - `base`: Set this to "models/StableDiffusionInpainting". This should point to the diffuser's inpainting model available [here](https://huggingface.co/runwayml/stable-diffusion-inpainting).
   - `vae`: Use "models/StableDiffusion". This must link to the original 1.5 stable diffusion model due to a diffusers issue. Get the vae from [here](https://huggingface.co/runwayml/stable-diffusion-v1-5).
   - `path`: Specify something like "models/DreamBooth_LoRA/realisticVisionV20_v20.inpainting.safetensors". It must be your Dreambooth model and needs to be an inpainting model. Note: You can convert existing models to inpainting models by adding the difference from the inpainting model and the standard model to any custom model.

3. **Execution**:
   ```bash
   python -m scripts.animate --config "configs/prompts/5-RealisticVision.yaml" --image_path "/images/image.png" --W 256 --H 256
   ```

## Considerations & Recommendations

- This model currently only works at roughly 256x256  resolutions. Retraining it to 512x512 didn't work for some reason, so you'd be best just upscaling with comfyui for now.
  
- In terms of making it work well, consider the prompts are not you telling the model what you want with this, you're guiding the generation for what is in the input image, if the input image and the prompts do not align, it will not work.

- You may have to try a few seeds per generation to get a nice image, it's a tiny bit unreliable.


## Acknowledgements
Special thanks to the developers behind AniamteDiff. https://github.com/guoyww/AnimateDiff.

