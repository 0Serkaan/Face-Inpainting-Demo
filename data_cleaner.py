import os

folder_path = "faces_highres1"
threshold = 5 * 1024  # 5 KB = 5120 bayt

deleted = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        size = os.path.getsize(file_path)
        if size < threshold:
            print(f"[SİLİNİYOR] {filename} ({size} bayt)")
            os.remove(file_path)
            deleted.append(filename)

if deleted:
    print(f"\nToplam {len(deleted)} dosya silindi:")
    for name in deleted:
        print(f" - {name}")
else:
    print("5 KB altı dosya bulunamadı.")
