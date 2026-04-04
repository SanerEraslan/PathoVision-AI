import os
import cv2
import numpy as np
from tqdm import tqdm

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Kaynak Klasörler
SRC_IMG_DIR = os.path.join(BASE_DIR, "dataset", "images")
SRC_MASK_DIR = os.path.join(BASE_DIR, "dataset", "masks")

# Hedef Klasörler
DEST_IMG_DIR = os.path.join(BASE_DIR, "dataset_tiled", "images")
DEST_MASK_DIR = os.path.join(BASE_DIR, "dataset_tiled", "masks")

PATCH_SIZE = 256
STRIDE = 128


# --- TÜRKÇE KARAKTER DOSTU OKUMA/YAZMA FONKSİYONLARI ---
def read_image_utf8(path, flag=cv2.IMREAD_COLOR):
    """
    Dosya yolunda Türkçe karakter olsa bile (Masaüstü vs.)
    dosyayı byte olarak okuyip OpenCV formatına çevirir.
    """
    try:
        # Dosyayı binary olarak oku
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            # OpenCV ile decode et
            return cv2.imdecode(numpy_array, flag)
    except Exception as e:
        # print(f"Okuma hatası: {path} -> {e}")
        return None


def write_image_utf8(path, img):
    """
    Dosya yolunda Türkçe karakter olsa bile kaydetmeyi sağlar.
    """
    try:
        # Uzantıyı al (.png)
        ext = os.path.splitext(path)[1]
        # Resmi encode et
        result, img_encoded = cv2.imencode(ext, img)
        if result:
            with open(path, "wb") as f:
                img_encoded.tofile(f)
            return True
    except Exception as e:
        print(f"Yazma hatası: {path} -> {e}")
    return False


def create_patches():
    # Hedef klasörleri oluştur
    if not os.path.exists(DEST_IMG_DIR):
        os.makedirs(DEST_IMG_DIR)
    if not os.path.exists(DEST_MASK_DIR):
        os.makedirs(DEST_MASK_DIR)

    # Dosya listesini al
    images = [f for f in os.listdir(SRC_IMG_DIR) if f.endswith(('.png', '.jpg', '.tif'))]
    print(f"✂️ {len(images)} adet ana görüntü parçalanıyor...")

    total_patches = 0

    for img_name in tqdm(images):
        img_path = os.path.join(SRC_IMG_DIR, img_name)
        mask_path = os.path.join(SRC_MASK_DIR, img_name)

        # YENİ FONKSİYONLARI KULLANIYORUZ
        img = read_image_utf8(img_path, cv2.IMREAD_COLOR)
        mask = read_image_utf8(mask_path, cv2.IMREAD_GRAYSCALE)  # 0 = Grayscale

        if img is None or mask is None:
            # Okunamayan dosyayı atla
            continue

        h, w, _ = img.shape

        # Padding (Eğer resim 256'dan küçükse tamamla)
        if h < PATCH_SIZE or w < PATCH_SIZE:
            pad_h = max(0, PATCH_SIZE - h)
            pad_w = max(0, PATCH_SIZE - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w, _ = img.shape

        # Parçalama Döngüsü
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                img_patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                # Boş maskeleri ele (Hızlandırma)
                if cv2.countNonZero(mask_patch) > 50:
                    patch_name = f"{os.path.splitext(img_name)[0]}_{y}_{x}.png"

                    # YENİ YAZMA FONKSİYONU
                    write_image_utf8(os.path.join(DEST_IMG_DIR, patch_name), img_patch)
                    write_image_utf8(os.path.join(DEST_MASK_DIR, patch_name), mask_patch)
                    total_patches += 1

    print(f"\n✅ PARÇALAMA TAMAMLANDI!")
    print(f"📊 Toplam Eğitim Verisi: {total_patches} adet.")
    print("🚀 Şimdi 'api/train.py' içinden DATA_DIR yolunu 'dataset_tiled' yapıp eğitime başlayabilirsin.")


if __name__ == "__main__":
    create_patches()