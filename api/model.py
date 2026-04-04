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
        
        # ÖNEMLİ: Eğer modeller api/models içindeyse bu kalsın. 
        # Eğer direkt api/ içindeyse os.path.join kısmını sil.
        self.models_dir = os.path.join(self.current_dir, "models")

        self.path_fast = os.path.join(self.models_dir, "evrensel_kanser_modeli.pth")
        self.path_pro = os.path.join(self.models_dir, "evrensel_kanser_modeli_pro.pth")

        self.models = {}

        if SMP_AVAILABLE:
            self._load_model("unet", self.path_fast)
            self._load_model("unetplus", self.path_pro)

    def _load_model(self, key, path):
        if os.path.exists(path):
            try:
                if key == "unet":
                    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
                else:
                    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
                
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models[key] = model.to(self.device)
                print(f"✅ Yüklendi: {key}")
            except Exception as e:
                print(f"❌ {key} yüklenemedi: {e}")
        else:
            print(f"⚠️ Dosya yok: {path}")

    def visualize_prediction(self, original_image, pred_mask):
        if not CV2_AVAILABLE: return original_image
        img_np = np.array(original_image.convert("RGB"))
        overlay = img_np.copy()
        overlay[pred_mask == 1] = [255, 0, 0] # Kanser
        overlay[pred_mask == 2] = [0, 255, 0] # Sağlıklı
        img_np = cv2.addWeighted(overlay, 0.4, img_np, 0.6, 0)
        return Image.fromarray(img_np)

    def predict(self, image_bytes, model_type="unet"):
        key = "unetplus" if model_type == "unetplusplus" else "unet"
        if key not in self.models:
            if self.models: key = list(self.models.keys())[0]
            else: return self._return_error_result(f"Model dosyası bulunamadı. Aranan konum: {self.models_dir}")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.models[key](input_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            mask_resized = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(image.size, Image.NEAREST))
            cancer_ratio = (np.sum(mask_resized == 1) / (mask_resized.size + 1e-6)) * 100

            return {
                "detected_cells": int(np.sum(mask_resized > 0) / 160),
                "predicted_ratio": round(cancer_ratio, 2),
                "visualization": self._image_to_base64(self.visualize_prediction(image, mask_resized)),
                "diagnosis": self._calculate_risk_status(cancer_ratio)
            }
        except Exception as e:
            return self._return_error_result(str(e))

    def _calculate_risk_status(self, ratio):
        if ratio < 2.0: return {"title": "DÜŞÜK RİSK", "color": "#10b981"}
        return {"title": "RİSKLİ", "color": "#dc2626"}

    def _return_error_result(self, msg):
        return {"diagnosis": {"title": "HATA", "message": msg, "color": "#dc2626"}}

    def _image_to_base64(self, image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
