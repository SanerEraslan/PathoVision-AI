import os
import torch
import sys


def check():
    print("🔍 SİSTEM TANI ARACI ÇALIŞIYOR...\n")
    print(f"🐍 Python Yolu: {sys.executable}")

    # 1. Kütüphane Kontrolü
    print("-" * 30)
    try:
        import segmentation_models_pytorch
        print("✅ Kütüphane DURUMU: YÜKLÜ (segmentation_models_pytorch)")
    except ImportError:
        print("❌ Kütüphane DURUMU: EKSİK! (Bilgisayarında bu kütüphane yok)")
        print("   -> Bu yüzden Pro mod 'Fallback' yapıyor ve sonuçlar bozuk çıkıyor.")
        print("   -> ÇÖZÜM: Terminale 'pip install segmentation-models-pytorch' yaz.")
        return  # Devam etmeye gerek yok

    # 2. Dosya Kontrolü
    print("-" * 30)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "unetplus_best.pth")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Dosya DURUMU: BULUNDU ({size_mb:.2f} MB)")

        if size_mb < 50:
            print("⚠️ UYARI: Dosya boyutu şüpheli derecede küçük! İndirme yarım kalmış olabilir.")
    else:
        print("❌ Dosya DURUMU: BULUNAMADI!")
        print(f"   Aranan Yol: {model_path}")
        return

    # 3. Yükleme Testi (En Kritik Kısım)
    print("-" * 30)
    print("⚙️  Model Yükleme Simülasyonu...")
    try:
        device = torch.device("cpu")
        # Mimariyi oluştur
        import segmentation_models_pytorch as smp
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)

        # Ağırlıkları yükle
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("\n🎉 TEBRİKLER! Dosya ve Kütüphane birbirine UYUMLU.")
        print("   Sorun kodda değil, belki önbellekte veya başka bir yerdedir.")

    except RuntimeError as e:
        print(f"\n❌ KRİTİK HATA: Mimari ve Dosya Uyuşmuyor!")
        print("   Bu dosya U-Net++ yapısında değil veya parametreler farklı.")
        print(f"   Hata: {e}")
    except Exception as e:
        print(f"\n❌ Beklenmeyen Hata: {e}")


if __name__ == "__main__":
    check()