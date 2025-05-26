import torch
import torch.nn as nn
from torchvision.models import vgg16

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, output, target):
        return self.loss(output, target)

class PerceptualLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vgg = vgg16(weights='IMAGENET1K_V1').features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self.device = device

    def forward(self, output, target):
        output = output.to(self.device)
        target = target.to(self.device)
        loss = 0.0
        for i in range(0, len(self.vgg), 4):
            out_feat = self.vgg[:i+4](output)
            tgt_feat = self.vgg[:i+4](target)
            loss += nn.functional.l1_loss(out_feat, tgt_feat)
        return loss

class TVLoss(nn.Module):
    def forward(self, x):
        h_var = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        w_var = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return h_var + w_var