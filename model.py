import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNetInpaint(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # 64
        self.enc3 = resnet.layer2                                         # 128
        self.enc4 = resnet.layer3                                         # 256
        self.enc5 = resnet.layer4                                         # 512

        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.upsample_out = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, masked, mask):
        x = torch.cat([masked, mask], dim=1)
        x = self.input_proj(x)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.final(d1)
        out = self.upsample_out(out)
        return out