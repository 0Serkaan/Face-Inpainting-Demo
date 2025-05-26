import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
from dataset import InpaintDataset
from model import ResNetInpaint
from losses import L1Loss, PerceptualLoss, TVLoss
from utils import save_checkpoint, save_sample

def train():
    device = DEVICE
    dataset = InpaintDataset(IMAGE_DIR, MASKED_DIR, MASK_DIR, IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = ResNetInpaint().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_l1 = L1Loss()
    loss_per = PerceptualLoss(device=device)
    loss_tv = TVLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        for masked, mask, gt in loader:
            masked, mask, gt = masked.to(device), mask.to(device), gt.to(device)
            output = model(masked, mask)

            l1 = loss_l1(output, gt)
            per = loss_per(output, gt)
            tv = loss_tv(output)

            loss = l1 + LAMBDA_PERCEPTUAL * per + LAMBDA_TV * tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

        if epoch % SAVE_SAMPLE_EVERY == 0:
            save_sample(output.detach(), masked.detach(), gt.detach(), epoch, OUTPUT_DIR)
            save_checkpoint(model, optimizer, epoch, os.path.join(CHECKPOINT_DIR, f"ckpt_{epoch}.pth"))

        if epoch % 5 == 0:
            mem = torch.cuda.memory_allocated() / 1024 ** 2
            print(f"[GPU] Allocated Memory: {mem:.2f} MB")

if __name__ == "__main__":
    train()