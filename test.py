import os
import torch
from PIL import Image
import torchvision.transforms as T
from model import ResNetInpaint

# Ayarlar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
CHECKPOINT_PATH = "checkpoints_last/ckpt_90.pth"
IMAGE_PATH = "ulas.jpg"
MASK_RADIUS = 20
SAVE_DIR = "test_outputs"
SAVE_PATH = os.path.join(SAVE_DIR, "result.png")
MASKED_PATH = os.path.join(SAVE_DIR, "masked_input.png")

# Dönüşümler
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

def create_center_mask(size, radius):
    import numpy as np
    import cv2
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(mask, center, radius, 255, -1)
    return torch.from_numpy(mask).unsqueeze(0).float() / 255.

def load_model():
    model = ResNetInpaint().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def run_inference():
    os.makedirs(SAVE_DIR, exist_ok=True)

    img = Image.open(IMAGE_PATH).convert("RGB")
    gt = tf(img).unsqueeze(0).to(DEVICE)

    mask = create_center_mask(IMG_SIZE, MASK_RADIUS).to(DEVICE)
    mask = mask.unsqueeze(0)  # [1, 1, H, W]
    masked = gt.clone() * (1 - mask)

    model = load_model()
    with torch.no_grad():
        output = model(masked, mask)

    from torchvision.utils import save_image
    # Sonuç karşılaştırma görseli: masked - output - gt
    save_image(torch.cat([masked, output, gt], dim=0), SAVE_PATH, nrow=1)

    # Maskelenmiş giriş görselini ayrı olarak kaydet
    save_image(masked, MASKED_PATH)

    print(f"✓ Sonuç kaydedildi: {SAVE_PATH}")
    print(f"✓ Maskelenmiş giriş görseli kaydedildi: {MASKED_PATH}")

if __name__ == "__main__":
    run_inference()
