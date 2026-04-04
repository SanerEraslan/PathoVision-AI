import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

# Kütüphane kontrolü
try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("⚠️ 'segmentation_models_pytorch' kütüphanesi eksik. Lütfen 'pip install segmentation-models-pytorch' yapın.")

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset_tiled")  # Tiled veri seti
MODELS_DIR = os.path.join(BASE_DIR, "models")

IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 50  # U-Net++ için 50 epoch harika sonuç verir
LR = 5e-5


# --- 1. ESKİ U-NET MİMARİSİ (Fast Mod İçin) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class CustomUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64);
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256);
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64);
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.down1(x);
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2));
        x4 = self.down4(self.pool(x3))
        x = self.up1(x4);
        x = self.conv1(torch.cat([x, x3], dim=1))
        x = self.up2(x);
        x = self.conv2(torch.cat([x, x2], dim=1))
        x = self.up3(x);
        x = self.conv3(torch.cat([x, x1], dim=1))
        return torch.sigmoid(self.final(x))


# --- VERİ SETİ ---
class CancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")

        if not os.path.exists(self.images_dir):
            self.images = []
        else:
            self.images = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))]
            print(f"📂 Veri seti: {len(self.images)} görsel.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 0)

        if self.transform:
            image = image.resize(IMG_SIZE);
            mask = mask.resize(IMG_SIZE)
            # Augmentation (Veri Çoğaltma)
            if random.random() > 0.5: image = TF.hflip(image); mask = TF.hflip(mask)
            if random.random() > 0.5: image = TF.vflip(image); mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.randint(-20, 20)
                image = TF.rotate(image, angle);
                mask = TF.rotate(mask, angle)
            return TF.to_tensor(image), TF.to_tensor(mask)
        return image, mask


# --- EĞİTİM FONKSİYONU ---
def train_model(model_name="unetplus"):  # Varsayılan artık U-Net++
    print(f"\n🚀 {model_name.upper()} Modeli Eğitiliyor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Cihaz: {device}")

    # 1. Model Seçimi
    model = None
    if model_name == "unet":
        model = CustomUNet().to(device)
        print("🏗️  Mimari: Standart U-Net")

    elif model_name == "unetplus":
        if not SMP_AVAILABLE:
            print("❌ HATA: Kütüphane eksik.")
            return
        print("🧬 Mimari: U-Net++ (Nested U-Net) - Pro Mod")
        # U-Net++ Modeli (Transfer Learning ile)
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",  # Ön eğitimli ağırlıklar (Hızlı öğrenir)
            in_channels=3,
            classes=1
        ).to(device)
    else:
        print(f"❌ Geçersiz model ismi: {model_name}");
        return

    # 2. Resume (Kaldığı yerden devam etme)
    save_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    if os.path.exists(save_path):
        print(f"🔄 Mevcut dosya bulundu, üzerine eğitim yapılacak: {save_path}")
        try:
            model.load_state_dict(torch.load(save_path))
        except:
            print("⚠️ Ağırlıklar uyumsuz, sıfırdan başlanıyor.")

    # 3. Hazırlık
    dataset = CancerDataset(DATA_DIR, transform=True)
    if len(dataset) == 0:
        print("❌ HATA: Veri seti boş. Lütfen 'create_patches.py' çalıştırın.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Loss Fonksiyonu: U-Net++ logits döndüğü için WithLogitsLoss şart
    if model_name == "unetplus":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCELoss()  # Custom U-Net sigmoid içeriyor

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Eğitim Döngüsü
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)

        # Her epoch sonu kaydet
        torch.save(model.state_dict(), save_path)
        print(f"💾 Kaydedildi. Ort. Kayıp: {avg_loss:.4f}")


if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

    # BURASI ARTIK UNETPLUS EĞİTİMİNİ BAŞLATIYOR
    train_model("unetplus")