import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
import os

# --- KÜTÜPHANE KONTROLLERİ ---
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

# AYARLAR
IMG_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelInference:
    def __init__(self):
        self.device = DEVICE
        # Dinamik yol tespiti
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Modellerin bulunduğu klasör (api/models)
        self.models_dir = os.path.join(self.current_dir, "models")

        # GÜNCEL MODEL İSİMLERİ
        self.path_fast = os.path.join(self.models_dir, "UNet_best.pth")
        self.path_pro = os.path.join(self.models_dir, "UNetPlusPlus_best.pth")

        self.models = {}

        if SMP_AVAILABLE:
            # 3 Sınıf (Background, Cancer, Healthy) desteğiyle yükleme
            self._load_model("unet", self.path_fast)
            self._load_model("unetplus", self.path_pro)

    def _load_model(self, key, path):
        if os.path.exists(path):
            try:
                # Modeller classes=3 (Multi-class) olarak yapılandırıldı
                if key == "unet":
                    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
                else:
                    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
                
                # Ağırlıkları yükle
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models[key] = model.to(self.device)
                print(f"✅ Yüklendi: {key} ({os.path.basename(path)})")
            except Exception as e:
                print(f"❌ {key} yüklenemedi: {e}")
        else:
            print(f"⚠️ Dosya bulunamadı: {path}")

    def visualize_prediction(self, original_image, pred_mask):
        if not CV2_AVAILABLE: return original_image
        
        img_np = np.array(original_image.convert("RGB"))
        overlay = img_np.copy()
        
        # pred_mask değerlerine göre renklendirme (0: BG, 1: Cancer, 2: Healthy)
        # Sınıf indeksleriniz farklıysa buradaki 1 ve 2'nin yerini değiştirebilirsiniz
        overlay[pred_mask == 1] = [231, 76, 60]  # Kanser: Kırmızı
        overlay[pred_mask == 2] = [46, 204, 113] # Sağlıklı: Yeşil
        
        # Orijinal görüntü ile maskeyi birleştir
        img_final = cv2.addWeighted(overlay, 0.4, img_np, 0.6, 0)
        return Image.fromarray(img_final)

    def predict(self, image_bytes, model_type="unet"):
        # Seçilen modele göre anahtarı belirle
        key = "unetplus" if model_type in ["unetplusplus", "unetplus"] else "unet"
        
        if key not in self.models:
            if self.models:
                key = list(self.models.keys())[0]
            else:
                return self._return_error_result(f"Model dosyaları bulunamadı. Lütfen {self.models_dir} klasörünü kontrol edin.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            orig_size = image.size
            
            # Ön İşleme
            transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            # AI Tahmini
            with torch.no_grad():
                output = self.models[key](input_tensor)
                # 3 kanallı çıktıdan en yüksek olasılıklı sınıfı seç (argmax)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # Maskeyi orijinal boyuta getir (Hızlı ve kesin olması için NEAREST kullanılır)
            mask_resized = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(orig_size, Image.NEAREST))
            
            # İstatistiksel Hesaplamalar
            total_pixels = mask_resized.size
            cancer_pixels = np.sum(mask_resized == 1) # Sınıf 1 = Kanser varsayıldı
            cancer_ratio = (cancer_pixels / (total_pixels + 1e-6)) * 100

            return {
                "detected_cells": int(np.sum(mask_resized > 0) / 160), # Tahmini hücre yoğunluğu
                "predicted_ratio": round(cancer_ratio, 2),
                "visualization": self._image_to_base64(self.visualize_prediction(image, mask_resized)),
                "diagnosis": self._calculate_risk_status(cancer_ratio)
            }
        except Exception as e:
            return self._return_error_result(f"Analiz sırasında hata: {str(e)}")

    def _calculate_risk_status(self, ratio):
        if ratio < 1.0: 
            return {"title": "TEMİZ / DÜŞÜK RİSK", "color": "#10b981", "status": "normal"}
        elif ratio < 5.0:
            return {"title": "ORTA RİSK - TAKİP ÖNERİLİR", "color": "#f59e0b", "status": "warning"}
        else:
            return {"title": "YÜKSEK RİSK / ŞÜPHELİ", "color": "#dc2626", "status": "critical"}

    def _return_error_result(self, msg):
        return {"diagnosis": {"title": "HATA", "message": msg, "color": "#dc2626"}}

    def _image_to_base64(self, image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
