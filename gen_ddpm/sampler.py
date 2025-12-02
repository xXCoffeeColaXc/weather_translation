import torch
import torchvision
import argparse
import yaml
import os

from dataclasses import dataclass
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
from gen_ddpm.model import UNet
from gen_ddpm.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class SamplerConfig:
    # train config
    sample_size: int = 2
    num_grid_rows: int = 2

    # model config
    im_channels: int = 3
    im_size: int = 128

    # diffusion config
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # folders
    model_path: str = 'gen_ddpm/checkpoints/1000-checkpoint.ckpt'
    samples: str = 'gen_ddpm/samples'


def save_images(xt, save_path, num_grid_rows):
    grid = make_grid(xt, nrow=num_grid_rows)
    img = torchvision.transforms.ToPILImage()(grid)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img.save(os.path.join(save_path,
                          f'old_x_{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}.png'))
    img.close()


def postprocess(xt, mean=[0.4865, 0.4998, 0.4323], std=[0.2326, 0.2276, 0.2659]):
    mean = torch.tensor(mean, device=xt.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=xt.device).view(1, -1, 1, 1)
    images = xt * std + mean
    images = (images * 255).clamp(0, 255).type(torch.uint8).detach().cpu()
    return images


def sample(model: UNet, scheduler: LinearNoiseScheduler, config: SamplerConfig, save_path: str = 'samples'):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """

    xt = torch.randn((config.sample_size, config.im_channels, config.im_size, config.im_size)).to(device)

    for i in tqdm(reversed(range(config.num_timesteps))):
        # Get prediction of noise
        t = torch.full((xt.size(0),), i, dtype=torch.long).to(device)

        # Predict noise
        with torch.no_grad():
            noise_pred = model(xt, scheduler.one_minus_cum_prod[t].view(-1, 1, 1, 1))

        mean, sigma, _ = scheduler.sample_prev_timestep2(xt, noise_pred, t)

        xt = mean + sigma if i != 0 else mean

    images = postprocess(xt)
    save_images(images, save_path, config.num_grid_rows)


def load_model(model_path: str) -> torch.nn.Module:
    model = UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_scheduler(num_timesteps: int, beta_start: float, beta_end: float) -> LinearNoiseScheduler:
    scheduler = LinearNoiseScheduler(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end)
    return scheduler


if __name__ == '__main__':
    config: SamplerConfig = SamplerConfig()
    # Load model with checkpoint
    model = load_model(config.model_path)

    # Load scheduler
    scheduler = load_scheduler(config.num_timesteps, config.beta_start, config.beta_end)

    sample(model, scheduler, config, config.samples)
