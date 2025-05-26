import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class InpaintDataset(Dataset):
    def __init__(self, image_dir, masked_dir, mask_dir, img_size=256):
        self.image_dir = image_dir
        self.masked_dir = masked_dir
        self.mask_dir = mask_dir
        self.filenames = os.listdir(image_dir)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        masked = Image.open(os.path.join(self.masked_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        img = self.transform(img)
        masked = self.transform(masked)
        mask = self.transform(mask)
        mask = (mask > 0.5).float()

        return masked, mask, img