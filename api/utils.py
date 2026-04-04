import numpy as np
import cv2
from PIL import Image
import io
import base64

def mask_to_base64(mask):
    """Mask'ı base64'e çevir"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', mask_uint8)
    return base64.b64encode(buffer).decode('utf-8')

def overlay_to_base64(image_bytes, mask, alpha=0.6):
    """Overlay görüntüsünü oluştur ve base64'e çevir"""
    try:
        # Görüntüyü yükle
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Mask'ı görüntü boyutuna getir
        if mask.shape != image_np.shape[:2]:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_resized = mask_pil.resize((image.width, image.height), Image.LANCZOS)
            mask = np.array(mask_resized) / 255.0
        
        # Overlay oluştur
        overlay = image_np.copy()
        
        # Hastalıklı bölgeleri kırmızı ile işaretle
        disease_mask = mask > 0.5
        overlay[disease_mask, 0] = np.clip(overlay[disease_mask, 0] + 100, 0, 255)  # Kırmızı artır
        overlay[disease_mask, 1] = overlay[disease_mask, 1] * 0.7  # Yeşili azalt
        overlay[disease_mask, 2] = overlay[disease_mask, 2] * 0.7  # Maviyi azalt
        
        # Alpha blending
        result = (1 - alpha) * image_np + alpha * overlay
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # PIL Image'a çevir ve base64'e encode et
        result_pil = Image.fromarray(result)
        buffer = io.BytesIO()
        result_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Overlay oluşturma hatası: {e}")
        # Hata durumunda orijinal görüntüyü döndür
        return base64.b64encode(image_bytes).decode('utf-8')

def overlay_mask_on_image(image_bytes, mask, alpha=0.5):
    """Mask'ı görüntü üzerine bindir (eski fonksiyon)"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            image = Image.open(io.BytesIO(image_bytes))
            image = np.array(image.convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Mask'ı resize et
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Kırmızı overlay
        overlay = image.copy()
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] + mask_resized * 255, 0, 255)
        
        result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        return result
    except:
        return None

def validate_image_format(filename):
    """Dosya formatını kontrol et"""
    if not filename:
        return False
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.heic', '.heif'}
    ext = filename.lower().split('.')[-1]
    return f'.{ext}' in allowed_extensions
