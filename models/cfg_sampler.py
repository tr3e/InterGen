import torch
import torch.nn as nn
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.s = cfg_scale

    def forward(self, x, timesteps, cond=None, mask=None):
        B, T, D = x.shape

        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)

        out = self.model(x_combined, timesteps_combined, cond=cond, mask=mask)

        out_cond = out[:B]
        out_uncond = out[B:]

        cfg_out = self.s *  out_cond + (1-self.s) *out_uncond
        return cfg_out
