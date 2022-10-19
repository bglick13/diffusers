from re import X
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import functions

from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


# Generation parameters
torch_device = "cpu"

scale = 3
h = 512
w = 512
ddim_steps = 45
ddim_eta = 0.0
torch.manual_seed(0)
to_tensor = transforms.ToTensor()

prompt = "A photo of Barack Obama smiling with a big grin"
input_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Official_portrait_of_Barack_Obama.jpg/440px-Official_portrait_of_Barack_Obama.jpg?20090114181817"
input_image = Image.open(requests.get(input_image_url, stream=True).raw).convert("RGB")
x = to_tensor(input_image).unsqueeze(0).to(torch_device)

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
scheduler = DDIMScheduler()
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

enc = vae.encode(x)
init_latent = enc.latent_dist.sample()
decoded_image = vae.decode(init_latent)

text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)
with torch.no_grad():
    original_text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
text_embeddings = original_text_embeddings.clone()
text_embeddings = functions.optimize_text_embeddings(
    text_embeddings,
    init_latent,
    unet,
    scheduler,
    torch_device=torch_device,
)
functions.finetune(text_embeddings, init_latent, unet, scheduler, torch_device=torch_device)
latents = functions.interpolate(
    original_text_embeddings,
    text_embeddings,
    unet,
    scheduler,
)
new_img = vae.decode(latents).image
