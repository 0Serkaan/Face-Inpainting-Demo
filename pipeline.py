import os
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ========================
# Pipeline: Dataset Hazırlık
# ========================
def prepare_dataset():
    raw_dir = "faces_highres1"
    output_root = "dataset_last"
    image_dir = os.path.join(output_root, "images")
    masked_dir = os.path.join(output_root, "masked")
    mask_dir = os.path.join(output_root, "masks")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    print("[1/2] Görseller dönüştürülüyor...")
    for filename in tqdm(os.listdir(raw_dir)):
        img_path = os.path.join(raw_dir, filename)
        image = Image.open(img_path).convert("RGB")
        resized = transform(image)
        T.ToPILImage()(resized).save(os.path.join(image_dir, filename))

    print("[2/2] Maskeleme işlemi başlatılıyor...")
    for filename in tqdm(os.listdir(image_dir)):
        img = cv2.imread(os.path.join(image_dir, filename))
        mask = np.zeros_like(img)

        for _ in range(np.random.randint(2, 5)):
            x, y = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
            r = np.random.randint(20, 40)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

        masked = img.copy()
        masked[mask == 255] = 0

        cv2.imwrite(os.path.join(mask_dir, filename), mask)
        cv2.imwrite(os.path.join(masked_dir, filename), masked)

    print("[✓] Dataset hazır: dataset/images, masked, masks")

if __name__ == "__main__":
    prepare_dataset()