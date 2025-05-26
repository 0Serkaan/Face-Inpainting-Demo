#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import gdown
import ijson
import requests
from tqdm import tqdm

def download_ffhq(num_images: int = 5000, output_dir: str = "faces_highres"):
    """
    gdown ile metadata JSON'unu indirir, ijson ile stream parse ederek
    ilk num_images yüz görsellerini faces_highres1/ altına kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Metadata JSON'unu Google Drive'dan indir
    #    ID = 16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA  (ffhq-dataset-v2.json)
    meta_path = "ffhq-dataset-v2.json"
    if not os.path.isfile(meta_path):
        print("Metadata JSON indiriliyor…")
        url = f"https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA"
        gdown.download(url, meta_path, quiet=False)

    # 2) JSON'u akışla parse ederek ve indirme döngüsü
    print(f"İlk {num_images} görsel için indirme başlıyor…")
    downloaded = 0
    with open(meta_path, "rb") as f:
        # Her kayıt: key -> value (value içinde image.file_url var)
        for _, record in ijson.kvitems(f, "", multiple_values=True):
            if downloaded >= num_images:
                break

            img_url = record["image"]["file_url"]
            ext = os.path.splitext(img_url)[1] or ".png"
            out_path = os.path.join(output_dir, f"{downloaded:05d}{ext}")

            try:
                resp = requests.get(img_url, stream=True, timeout=30)
                resp.raise_for_status()
                with open(out_path, "wb") as f_img:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f_img.write(chunk)
                downloaded += 1
            except Exception as e:
                print(f"[{downloaded}] İndirme hatası: {e}")
                # hata varsa atla, bir sonraki kayda geç

    print(f"Tamamlandı! '{output_dir}' klasöründe {downloaded} görsel var.")

if __name__ == "__main__":

    download_ffhq(num_images=50000, output_dir="faces_highres1")
