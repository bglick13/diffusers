import torch
from tqdm import tqdm


def optimize_text_embeddings(
    emb, init_latent, unet, scheduler, batch_size=1, num_steps=50, height=512, width=512, torch_device="cpu"
):
    emb.requires_grad = True
    lr = 0.001
    it = 500
    opt = torch.optim.Adam([emb], lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    pbar = tqdm(range(it))
    for i in pbar:
        opt.zero_grad()

        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(1000, (1,), device=torch_device)
        # predict the residual
        z = scheduler.add_noise(init_latent.detach(), noise, t_enc)
        noise_pred = unet(z, t_enc, encoder_hidden_states=emb).sample

        # noise_pred = scheduler.step(noise_pred, t_enc, noise).prev_sample
        # latents = scheduler.step(noise_pred, t_enc, noise).prev_sample

        loss = criteria(noise_pred, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()
    return emb


def finetune(emb, init_latent, unet, scheduler, torch_device="cpu"):
    emb.requires_grad = False
    unet.train()
    lr = 1e-6
    it = 1000
    opt = torch.optim.Adam(unet.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    history = []
    pbar = tqdm(range(it))
    for i in pbar:
        opt.zero_grad()

        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(range(len(scheduler.timesteps)), (1,), device=torch_device)
        z = scheduler.add_noise(init_latent.detach(), noise, t_enc)
        noise_pred = unet(init_latent, t_enc, encoder_hidden_states=emb).sample

        loss = criteria(noise_pred, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()


def interpolate(
    original_text_embeddings,
    emb,
    unet,
    scheduler,
    torch_device="cpu",
    alpha=0.9,
    batch_size=1,
    height=512,
    width=512,
    scale=3,
):
    new_emb = alpha * original_text_embeddings + (1 - alpha) * emb
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
    )
    latents = latents.to(torch_device)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=new_emb).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents
