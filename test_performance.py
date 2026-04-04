import torch
import numpy as np
import os
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms

# --- AYARLAR ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = r"C:\Users\Saner Eraslan\Desktop\b\models"
TEST_IMG_DIR = "test_images/"
TEST_MASK_DIR = "test_masks/"

model_files = {
    "Standart (U-Net)": "evrensel_kanser_modeli.pth",
    "Pro (U-Net++)": "evrensel_kanser_modeli_pro.pth"
}


def calculate_optimized_metrics(pred, target):
    smooth = 1e-6

    # --- [ADIM 1] MORFOLOJİK İYİLEŞTİRME ---
    # Modelin ham çıktısını (0,1,2) binary formata getir ve temizle
    pred_binary = ((pred == 1) | (pred == 2)).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    # Opening: Küçük gürültü piksellerini yok eder
    pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, kernel)
    # Closing: Hücre içindeki küçük siyah noktaları doldurur
    pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)

    # --- GERÇEK MASKEYİ HAZIRLA ---
    if len(target.shape) == 3:
        target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    target_binary = (target > 0).astype(np.float32)

    # --- HESAPLAMA ---
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)

    if np.sum(target_binary) == 0:
        return (1.0, 1.0) if np.sum(pred_binary) == 0 else (0.0, 0.0)

    dice = (2. * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)

    return dice, iou


def load_model(name, filename):
    path = os.path.join(MODELS_DIR, filename)
    print(f"🏗️ {name} yükleniyor...")
    if "pro" in filename.lower():
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3).to(DEVICE)
    else:
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# --- TEST DÖNGÜSÜ ---
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
test_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

results_summary = {}

for name, filename in model_files.items():
    model = load_model(name, filename)
    all_dice, all_iou = [], []

    for img_name in test_files:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        mask_path = os.path.join(TEST_MASK_DIR, img_name)

        if os.path.exists(mask_path):
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            # --- [ADIM 2] TTA (TEST TIME AUGMENTATION) ---
            with torch.no_grad():
                # 1. Orijinal Görüntü Tahmini
                out_normal = model(input_tensor)

                # 2. Yatay Çevrilmiş Görüntü Tahmini
                out_flipped = model(torch.flip(input_tensor, [3]))
                out_flipped = torch.flip(out_flipped, [3])  # Tahmini geri çevir

                # İki tahminin ortalamasını alarak daha kararlı bir sonuç elde et
                output = (out_normal + out_flipped) / 2
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # Maskeyi Oku ve Boyutlandır
            true_mask_raw = np.array(Image.open(mask_path))
            true_mask_resized = cv2.resize(true_mask_raw, (256, 256), interpolation=cv2.INTER_NEAREST)

            dice, iou = calculate_optimized_metrics(pred_mask, true_mask_resized)
            all_dice.append(dice)
            all_iou.append(iou)

    results_summary[name] = {"Dice": np.mean(all_dice) * 100, "IoU": np.mean(all_iou) * 100}

# --- SONUÇ PANELİ ---
print("\n" + "X" * 45)
print("🔬 PATHVISION AI: OPTİMİZE EDİLMİŞ DOĞRULUK RAPORU")
print("X" * 45)
for m_name, metrics in results_summary.items():
    print(f"{m_name}:")
    print(f"   >> Dice (Doğruluk): %{metrics['Dice']:.2f}")
    print(f"   >> IoU (Kesişim): %{metrics['IoU']:.2f}")
    print("-" * 30)