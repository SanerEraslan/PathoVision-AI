import os
import shutil
from tqdm import tqdm

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Hedef Klasör (Senin mevcut dataset'in)
DEST_IMG_DIR = os.path.join(BASE_DIR, "dataset", "images")
DEST_MASK_DIR = os.path.join(BASE_DIR, "dataset", "masks")

# 2. Kaynak Klasör (Yeni indirdiğin MoNuSeg verisi)
SOURCE_DIR = os.path.join(BASE_DIR, "raw_data_monuseg")


def find_folder(root_folder, target_names):
    """
    Alt klasörlerde derinlemesine arama yapar.
    target_names: Aranacak klasör isimleri listesi (örn: ['Tissue Images', 'Images'])
    """
    for root, dirs, files in os.walk(root_folder):
        for d in dirs:
            if d in target_names:
                return os.path.join(root, d)
    return None


def merge_monuseg():
    print("🔄 MoNuSeg Veri Seti Taranıyor ve Entegre Ediliyor...")

    # Klasörleri otomatik bul (Derinlemesine arama)
    print("🔎 Klasörler aranıyor...")

    # Olası klasör isimleri (Kaggle versiyonlarına göre)
    src_img_path = find_folder(SOURCE_DIR, ["Tissue Images", "Images", "images"])
    src_mask_path = find_folder(SOURCE_DIR, ["Ground Truth", "Masks", "masks", "Binary Masks"])

    if not src_img_path or not src_mask_path:
        print("\n❌ HATA: MoNuSeg klasörleri bulunamadı!")
        print(f"   Aranan yer: {SOURCE_DIR}")
        print("   Lütfen zip dosyasını 'raw_data_monuseg' klasörüne çıkardığınızdan emin olun.")
        return

    print(f"✅ Görüntü Klasörü Bulundu: {src_img_path}")
    print(f"✅ Maske Klasörü Bulundu:  {src_mask_path}")

    # Dosyaları Taşı
    files = os.listdir(src_img_path)
    count = 0

    print("📦 Dosyalar kopyalanıyor...")
    for f in tqdm(files):
        if f.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
            # Orijinal resim
            original_img = os.path.join(src_img_path, f)

            # Maske eşleştirmesi
            # MoNuSeg'de genelde .tif resimlerin maskesi .png veya .xml olur.
            # Kaggle versiyonunda genelde aynı isimdedir ama uzantı .png'dir.
            base_name = os.path.splitext(f)[0]

            # Olası maske isimleri
            possible_masks = [
                f,  # ayni_isim.png
                base_name + ".png",  # ayni_isim.png
                base_name + ".tif",  # ayni_isim.tif
                base_name + "_mask.png"  # ayni_isim_mask.png
            ]

            original_mask = None
            for pm in possible_masks:
                candidate = os.path.join(src_mask_path, pm)
                if os.path.exists(candidate):
                    original_mask = candidate
                    break

            if original_mask and os.path.exists(original_img):
                # Çakışmayı önlemek için isme 'monuseg_' ekle
                # Tüm çıktıları .png yapalım standart olsun
                new_filename = f"monuseg_{base_name}.png"

                # Resmi kopyala (Gerekirse dönüştür)
                try:
                    shutil.copy(original_img, os.path.join(DEST_IMG_DIR, new_filename))
                    shutil.copy(original_mask, os.path.join(DEST_MASK_DIR, new_filename))
                    count += 1
                except Exception as e:
                    print(f"Hata: {e}")

    print(f"\n✅ İŞLEM BAŞARILI!")
    print(f"➕ {count} adet yeni MoNuSeg verisi projeye eklendi.")

    if count > 0:
        print("💡 İPUCU: Şimdi 'create_patches.py' (Tiling) kodunu çalıştırırsan")
        print("   bu büyük resimlerden binlerce yeni eğitim verisi çıkacaktır.")


if __name__ == "__main__":
    if not os.path.exists(DEST_IMG_DIR):
        print("⚠️ Önce prepare_data.py çalıştırıp ana dataseti oluşturmalısın.")
    else:
        merge_monuseg()