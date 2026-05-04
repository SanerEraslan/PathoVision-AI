import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
import os
from huggingface_hub import hf_hub_download

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

IMG_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelInference:
    def __init__(self):
        self.device = DEVICE
        self.repo_id = "SanerEraslan/PathoVision-Models" # Kendi Repo ID'n
        self.models = {}
        
        # Hugging Face üzerindeki dosya isimlerin
        self.model_files = {
            "unet": "UNet_best.pth",
            "unetplus": "UNetPlusPlus_best.pth"
        }

        if SMP_AVAILABLE:
            self._load_all_models()

    def _load_all_models(self):
        for key, filename in self.model_files.items():
            try:
                # Modeli Hugging Face'ten indir
                checkpoint_path = hf_hub_download(repo_id=self.repo_id, filename=filename)
                
                # Model mimarisini oluştur (classes=1 hatayı çözen kısımdır)
                if key == "unet":
                    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
                else:
                    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
                
                # Ağırlıkları yükle
                model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                model.eval()
                self.models[key] = model.to(self.device)
                print(f"✅ HF üzerinden yüklendi: {filename}")
            except Exception as e:
                print(f"❌ {filename} yüklenemedi: {e}")

    def predict(self, image_bytes, model_type="unet"):
        key = "unetplus" if "plus" in model_type.lower() else "unet"
        if key not in self.models:
            return {"diagnosis": {"title": "HATA", "message": "Model yüklenemedi.", "color": "#dc2626"}}

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            orig_size = image.size
            
            transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.models[key](input_tensor)
                # Binary model olduğu için Sigmoid kullanıyoruz
                prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()

            # Maskeyi orijinal boyuta getir
            mask_resized = np.array(Image.fromarray((prob_mask * 255).astype(np.uint8)).resize(orig_size, Image.BILINEAR)) / 255.0
            
            # Arkaplan ayıklama (Beyaz alanlar)
            grayscale = np.mean(np.array(image), axis=2)
            is_bg = grayscale > 230
            
            # Kanserli alan hesabı (Eşik: 0.5)
            is_cancer = (mask_resized > 0.5) & (~is_bg)
            cancer_ratio = (np.sum(is_cancer) / (mask_resized.size + 1e-6)) * 100

            return {
                "predicted_ratio": round(cancer_ratio, 2),
                "visualization": self._image_to_base64(self.visualize_prediction(image, is_cancer, is_bg)),
                "diagnosis": self._calculate_risk_status(cancer_ratio),
                "stats": {
                    "Kanserli": round(cancer_ratio, 1),
                    "Sağlıklı": round(100 - cancer_ratio - (np.sum(is_bg)/is_bg.size*100), 1),
                    "Arkaplan": round(np.sum(is_bg)/is_bg.size*100, 1)
                }
            }
        except Exception as e:
            return {"diagnosis": {"title": "HATA", "message": str(e), "color": "#dc2626"}}

    def visualize_prediction(self, original_image, cancer_mask, bg_mask):
        img_np = np.array(original_image.convert("RGB"))
        overlay = img_np.copy()
        
        # Renklendirme
        overlay[cancer_mask] = [231, 76, 60] # Kanser: Kırmızı
        overlay[bg_mask] = [189, 195, 199]   # Arkaplan: Gri
        
        if CV2_AVAILABLE:
            img_final = cv2.addWeighted(overlay, 0.4, img_np, 0.6, 0)
            return Image.fromarray(img_final)
        return Image.fromarray(overlay)

    def _calculate_risk_status(self, ratio):
        if ratio < 1.0: return {"title": "DÜŞÜK RİSK", "color": "#10b981"}
        return {"title": "YÜKSEK RİSK", "color": "#dc2626"}

    def _image_to_base64(self, image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
