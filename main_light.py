import torch
import argparse
import pandas as pd
import sys

from nerf.provider import NeRFDataset
from nerf.utils import *

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F


# torch.autograd.set_detect_anomaly(True)

class Options:
    def __init__(self, text=None, workspace='workspace'):
        self.file = None
        self.text = text
        self.negative = ''
        self.fp16 = True
        self.cuda_ray = True
        self.eval_interval = 1
        self.test_interval = 100
        self.workspace = workspace
        self.seed = None
        self.image = None
        self.image_config = None
        self.known_view_interval = 4
        self.IF = False
        self.guidance = ['SD']
        self.guidance_scale = 100
        self.save_mesh = False
        self.mcubes_resolution = 256
        self.decimate_target = 5e4
        self.dmtet = False
        self.tet_grid_size = 128
        self.init_with = ''
        self.lock_geo = False
        self.perpneg = False
        self.negative_w = -2
        self.front_decay_factor = 2
        self.side_decay_factor = 10
        # self.iters = 10000
        self.iters = 5000
        self.lr = 1e-3
        self.ckpt = 'latest'
        self.cuda_ray = False
        self.taichi_ray = False
        self.max_steps = 1024
        self.num_steps = 64
        self.upsample_steps = 32
        self.update_extra_interval = 16
        self.max_ray_batch = 4096
        self.latent_iter_ratio = 0.2
        self.albedo_iter_ratio = 0
        self.min_ambient_ratio = 0.1
        self.textureless_ratio = 0.2
        self.jitter_pose = False
        self.jitter_center = 0.2
        self.jitter_target = 0.2
        self.jitter_up = 0.02
        self.uniform_sphere_rate = 0
        self.grad_clip = -1
        self.grad_clip_rgb = -1
        self.bg_radius = 1.4
        self.density_activation = 'exp'
        self.density_thresh = 10
        self.blob_density = 5
        self.blob_radius = 0.2
        self.backbone = 'grid'
        self.optim = 'adan'
        self.sd_version = '2.1'
        self.hf_key = None
        self.fp16 = False
        self.vram_O = False
        self.w = 64
        self.h = 64
        self.known_view_scale = 1.5
        self.known_view_noise_scale = 2e-3
        self.dmtet_reso_scale = 8
        self.batch_size = 1
        self.bound = 1
        self.dt_gamma = 0
        self.min_near = 0.01
        self.radius_range = [3.0, 3.5]
        self.theta_range = [45, 105]
        self.phi_range = [-180, 180]
        self.fovy_range = [10, 30]
        self.default_radius = 3.2
        self.default_polar = 90
        self.default_azimuth = 0
        self.default_fovy = 20
        self.progressive_view = False
        self.progressive_view_init_ratio = 0.2
        self.progressive_level = False
        self.angle_overhead = 30
        self.angle_front = 60
        self.t_range = [0.02, 0.98]
        self.dont_override_stuff = False
        self.lambda_entropy = 1e-3
        self.lambda_opacity = 0
        self.lambda_orient = 1e-2
        self.lambda_tv = 0
        self.lambda_wd = 0
        self.lambda_mesh_normal = 0.5
        self.lambda_mesh_laplacian = 0.5
        self.lambda_guidance = 1
        self.lambda_rgb = 1000
        self.lambda_mask = 500
        self.lambda_normal = 0
        self.lambda_depth = 10
        self.lambda_2d_normal_smooth = 0
        self.lambda_3d_normal_smooth = 0
        self.save_guidance = False
        self.save_guidance_interval = 10
        self.gui = False
        self.W = 800
        self.H = 800
        self.radius = 5
        self.fovy = 20
        self.light_theta = 60
        self.light_phi = 0
        self.max_spp = 1
        self.zero123_config = './pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml'
        self.zero123_ckpt = 'pretrained/zero123/zero123-xl.ckpt'
        self.zero123_grad_scale = 'angle'
        self.dataset_size_train = 100
        self.dataset_size_valid = 8
        self.dataset_size_test = 100
        self.exp_start_iter = None
        self.exp_end_iter = None


class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
        pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)


        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import time
    start_time = time.time()
    
    # See https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
    text = "a hamburger"
    workspace = "trial_7"
    
    # Options 인스턴스 생성
    opt = Options(text=text, workspace=workspace)

    opt.seed = 5

    opt.images, opt.ref_radii, opt.ref_polars, opt.ref_azimuths, opt.zero123_ws = [], [], [], [], []
    opt.default_zero123_w = 1

    opt.exp_start_iter = opt.exp_start_iter or 0
    opt.exp_end_iter = opt.exp_end_iter or opt.iters

    opt.images = None

    # from nerf.network_grid import NeRFNetwork
    from nerf.network import NeRFNetwork
    print(opt)

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt).to(device)

    print(model)

    train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

    from optimizer import Adan
    # Adan usually requires a larger LR
    optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)

    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    guidance = nn.ModuleDict()

    # from guidance.sd_utils import StableDiffusion
    
    guidance['SD'] = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range)

    trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)

    trainer.default_view_data = train_loader._data.get_default_view_data()

    valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
    test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)

    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    trainer.train(train_loader, valid_loader, test_loader, max_epoch)

    trainer.save_mesh()
    
    
    end_time = time.time()
    # 소요된 시간 계산 (초 단위)
    elapsed_time = end_time - start_time

    # 소요된 시간을 분 단위로 변환
    elapsed_minutes = elapsed_time / 60

    print(f"소요된 시간: {elapsed_minutes:.2f} 분")

