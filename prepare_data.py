import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

# --- AYARLAR ---
# İndirdiğin zipi çıkardığın yer (Burayı kendi bilgisayarına göre düzenle!)
# Eğer proje klasörü içinde 'raw_data' diye klasör açıp oraya çıkardıysan dokunmana gerek yok.
RAW_DATA_PATH = "raw_data"

# Hedef Klasörler (Bizim projenin dataset klasörü)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_IMAGES = os.path.join(BASE_DIR, "dataset", "images")
DEST_MASKS = os.path.join(BASE_DIR, "dataset", "masks")


def prepare_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ HATA: '{RAW_DATA_PATH}' klasörü bulunamadı!")
        print("Lütfen indirdiğiniz stage1_train.zip dosyasını 'raw_data' klasörüne çıkarın.")
        return

    # Klasörleri temizle ve oluştur
    if os.path.exists(DEST_IMAGES): shutil.rmtree(DEST_IMAGES)
    if os.path.exists(DEST_MASKS): shutil.rmtree(DEST_MASKS)
    os.makedirs(DEST_IMAGES)
    os.makedirs(DEST_MASKS)

    # Klasör listesini al
    ids = next(os.walk(RAW_DATA_PATH))[1]
    print(f"📦 Toplam {len(ids)} adet görüntü işleniyor...")

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        path = os.path.join(RAW_DATA_PATH, id_)

        # 1. Orijinal Resmi Al
        # Genelde /images/ID.png şeklindedir
        img_folder = os.path.join(path, "images")
        if not os.path.exists(img_folder): continue

        img_filename = os.listdir(img_folder)[0]
        img = Image.open(os.path.join(img_folder, img_filename)).convert("RGB")

        # 2. Maskeleri Birleştir (Merging)
        mask_folder = os.path.join(path, "masks")
        mask = np.zeros((img.height, img.width), dtype=np.uint8)

        # O klasördeki tüm maske parçalarını üst üste ekle
        for mask_file in os.listdir(mask_folder):
            mask_part = np.array(Image.open(os.path.join(mask_folder, mask_file)).convert("L"))
            mask = np.maximum(mask, mask_part)  # Mantıksal OR işlemi gibi birleştir

        # 3. Kaydet (İsimleri basitleştir: 0.png, 1.png...)
        # Dosya boyutlarını sabitlemek istersen resize eklenebilir ama train.py zaten yapıyor.
        final_name = f"{id_}.png"

        img.save(os.path.join(DEST_IMAGES, final_name))
        Image.fromarray(mask).save(os.path.join(DEST_MASKS, final_name))

    print(f"\n✅ Veri hazırlığı tamamlandı!")
    print(f"📂 Görüntüler: {DEST_IMAGES}")
    print(f"📂 Maskeler: {DEST_MASKS}")
    print("🚀 Artık 'python train.py' komutunu çalıştırabilirsin.")


if __name__ == "__main__":
    prepare_dataset()