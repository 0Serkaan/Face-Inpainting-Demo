import os
import torch
from torchvision.utils import save_image

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)

def save_sample(output, masked, gt, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    grid = torch.cat([masked, output, gt], dim=0)
    save_image(grid, os.path.join(output_dir, f"sample_epoch_{epoch}.png"), nrow=masked.size(0))