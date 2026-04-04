# Kanser Hücresi Tespit Sistemi

Hastalıklı hücrelerin tespiti ve segmentasyonu için U-Net tabanlı deep learning modeli.

## Özellikler

- ✅ Hastalıklı bölge segmentasyonu
- ✅ Hastalık oranı tahmini (sağlıklı/hastalıklı hücre oranı)
- ✅ Güven skoru hesaplama
- ✅ Çoklu format desteği: `.png`, `.jpg`, `.jpeg`, `.tif`, `.heic`
- ✅ REST API


## Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Sunucuyu başlat (Otomatik tarayıcı açılır)
python start_server.py

# Veya manuel olarak
cd api
python app.py
```

## Kullanım

### Web Arayüzü

1. Sunucuyu başlatın: `python start_server.py`
2. Tarayıcınızda http://localhost:8000 adresine gidin
3. Görüntü dosyasını sürükleyip bırakın veya seçin
4. "Analiz Et" butonuna tıklayın
5. Sonuçları görüntüleyin

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Tahmin
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.tif"
```

Response:
```json
{
  "success": true,
  "filename": "image.tif",
  "results": {
    "detected_cells": 245,
    "total_cells": 756,
    "detection_ratio": 0.324,
    "confidence_score": 0.892,
    "segmentation_mask": "base64_encoded_mask"
  }
}
```

#### 3. Maske ile Tahmin
```bash
curl -X POST "http://localhost:8000/predict_with_overlay" \
  -F "file=@image.tif"
```

Response:
```json
{
  "success": true,
  "filename": "image.tif",
  "results": {
    "detected_cells": 245,
    "total_cells": 756,
    "detection_ratio": 0.324,
    "confidence_score": 0.892,
    "mask_image": "base64_encoded_mask"
  }
}
```

### Python Client Örneği

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("image.tif", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Tespit Edilen Hücre: {result['results']['detected_cells']}")
print(f"Toplam Hücre: {result['results']['total_cells']}")
print(f"Tespit Oranı: {result['results']['detection_ratio']:.2%}")
print(f"Güven Skoru: {result['results']['confidence_score']:.2%}")
```

## Proje Yapısı

```
NuSeC/
├── api/
│   ├── app.py             # FastAPI uygulaması
│   ├── model.py           # Model inference
│   └── utils.py           # Yardımcı fonksiyonlar
├── frontend/
│   └── index.html         # Web arayüzü
├── models/                # Eğitilmiş modeller
│   └── best_model.pth
├── data/                  # Dataset
│   ├── train/
│   ├── val/
│   └── test/
├── start_server.py        # Sunucu başlatıcı
└── requirements.txt
```

## Model Detayları

- **Mimari**: U-Net + ResNet34 encoder
- **Input**: 512x512 RGB görüntü
- **Output**: 
  - Segmentation mask (512x512)
  - Tespit edilen hücre sayısı
  - Toplam hücre sayısı

## API Dokümantasyonu

API çalıştıktan sonra:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Notlar

- Model dosyası ~85MB
- Inference süresi: ~100ms (CPU), ~20ms (GPU)


.
.
