import math

import torch
import torch.nn.functional as F
from einops import reduce
from Models.interpretable_diffusion.gaussian_diffusion import (
    DiffusionWL as _BaseDiffusionWL,
)
from Models.interpretable_diffusion.model_utils import default, extract
from tqdm.auto import tqdm


class DiffusionWL(_BaseDiffusionWL):
    """Extends DiffusionWL with progressive distillation support.

    The teacher model guides a student with half the diffusion steps,
    following "Progressive Distillation for Fast Sampling of Diffusion
    Models" (Salimans & Ho, 2022).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set by the distillation loop in Engine/trainer.py
        self.oriteacher = None   # original full-step teacher
        self.orinumtimesteps = None
        self.count = 0           # distillation generation index
        self.target = None

    @torch.no_grad()
    def sample(self, shape):
        """DDPM sampling using sampling_timesteps (halved each distillation step)."""
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.sampling_timesteps)),
            desc="sampling loop time step",
            total=self.sampling_timesteps,
        ):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        """DDIM-style accelerated sampling."""
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def generate_data(self, x_start, noise=None):
        """Generate teacher and student predictions at the same noisy input."""
        b, device = x_start.shape[0], x_start.device
        t = torch.randint(
            0, self.teacher.model.num_timesteps, (b,), device=device
        ).long()
        noise = default(noise, lambda: torch.randn_like(x_start))
        zt = self.teacher.model.q_sample(x_start=x_start, t=t, noise=noise)
        x_teacher = self.teacher.model.output(zt, t, padding_masks=None)
        ratio = self.teacher.model.num_timesteps // self.num_timesteps
        x_student = self.output(zt, t // ratio, padding_masks=None)
        return x_teacher, x_student

    def distill_target(self, x_start, t_s, t, teacher, noise=None):
        """Compute the distillation target from the original (full-step) teacher."""
        if teacher is None:
            raise ValueError("Teacher is not defined")
        noise = default(noise, lambda: torch.randn_like(x_start))
        zt = self.oriteacher.model.q_sample(x_start=x_start, t=t, noise=noise)
        x_tilda = teacher.model.output(zt, t_s, padding_masks=None)
        return x_tilda, zt

    def _distill_loss(self, x_start, t, target=None, padding_masks=None):
        count = self.count
        t_s = t // (2 ** (count + 1))
        t_s[t_s == self.num_timesteps] -= 1
        t_s_old = t if count == 0 else t // (2**count)

        target, zt = self.distill_target(
            x_start=x_start, t_s=t_s_old, t=t, teacher=self.teacher
        )
        student_out = self.output(zt, t_s, padding_masks)

        distill_loss = self.loss_fn(student_out, target, reduction="none")

        if self.use_ff:
            fft1 = torch.fft.fft(student_out.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(target.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(
                torch.real(fft1), torch.real(fft2), reduction="none"
            ) + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction="none")
            distill_loss += self.ff_weight * fourier_loss

        distill_loss = reduce(distill_loss, "b ... -> b (...)", "mean")
        distill_loss = distill_loss * extract(self.loss_weight, t_s, distill_loss.shape)
        return distill_loss.mean()

    def forward(self, x, **kwargs):
        b, n, device, feature_size = (
            x.shape[0],
            x.shape[2],
            x.device,
            self.feature_size,
        )
        assert n == feature_size, f"number of variable must be {feature_size}"
        if self.oriteacher is None:
            t = 2 * torch.randint(0, self.num_timesteps, (b,), device=device).long()
        else:
            t = torch.randint(0, self.orinumtimesteps, (b,), device=device).long()
        return self._distill_loss(x_start=x, t=t, **kwargs)
