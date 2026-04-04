import streamlit as st
import os
import io
import sys
from PIL import Image

# Alt klasördeki model.py'yi görebilmesi için path ekleyelim
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ModelInference

# 1. SAYFA AYARLARI
st.set_page_config(
    page_title="PathoVision AI - Kanser Hücresi Tespit",
    page_icon="🔬",
    layout="wide"
)

# 2. ÖZEL TASARIM (CSS)
def apply_custom_design():
    st.markdown("""
        <style>
        .block-container { padding: 0rem 3rem; }
        .stApp { background-color: #f4f7f6; }
        .custom-header {
            background: linear-gradient(135deg, #1e2b3c 0%, #2d3e50 100%);
            padding: 40px; border-radius: 0 0 20px 20px;
            color: white; text-align: center; margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .analysis-card {
            background: white; padding: 20px; border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;
        }
        .stButton>button {
            background: linear-gradient(to right, #4CAF50, #45a049);
            color: white; border-radius: 10px; width: 100%;
            height: 3.5em; font-size: 18px; font-weight: bold;
        }
        </style>
        <div class="custom-header">
            <h1 style="margin:0;">🔬 PathoVision AI</h1>
            <p>Gelişmiş Histopatolojik Analiz Sistemi</p>
        </div>
    """, unsafe_allow_html=True)

apply_custom_design()

# 3. MODEL SERVİSİNİ BAŞLAT (Hata Ayıklama Moduyla)
@st.cache_resource
def load_model_service():
    try:
        # Klasör varlık kontrolü
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "models")):
            st.error("Kritik Hata: 'api/models' klasörü bulunamadı!")
        return ModelInference()
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None

model_service = load_model_service()

# 4. ARAYÜZ BÖLÜMLERİ
with st.sidebar:
    st.header("⚙️ Ayarlar")
    model_type = st.selectbox("Model Seçimi", ["unet", "unetplusplus"])
    st.info("Sistem Durumu: Çevrimiçi 🟢")

st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Görüntü Yükle", type=["jpg", "png", "tif"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file and model_service:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Kaynak Görüntü", use_container_width=True)
    
    with col2:
        if st.button("Analizi Başlat"):
            with st.spinner('İşleniyor...'):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                results = model_service.predict(img_byte_arr.getvalue(), model_type=model_type)
                
                if results:
                    st.success("Tamamlandı!")
                    st.metric("Hücre Sayısı", results.get("detected_cells", 0))
                    st.json(results)
