import streamlit as st
import os
import io
import sys
from PIL import Image

# 1. YOL AYARLARI (Modül ve Path Çözümleri)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- Mailer Fonksiyonu Entegrasyonu ---
try:
    from mailer import send_pathovision_report
except ImportError:
    try:
        from api.mailer import send_pathovision_report
    except ImportError:
        st.error("Kritik Hata: mailer.py bulunamadı. Lütfen dosya dizinini kontrol edin.")

# Model servisi
from model import ModelInference

# 2. SAYFA KONFİGÜRASYONU
st.set_page_config(
    page_title="PathoVision AI - Kanser Hücresi Tespit",
    page_icon="🔬",
    layout="wide"
)


# 3. ÖZEL CSS TASARIMI
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


# 4. MODEL YÜKLEME
@st.cache_resource
def load_model_service():
    try:
        return ModelInference()
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None


model_service = load_model_service()

# 5. YAN MENÜ (SIDEBAR)
with st.sidebar:
    st.header("⚙️ Ayarlar")
    model_type = st.selectbox("Model Seçimi", ["unet", "unetplusplus"])
    st.info("Sistem Durumu: Çevrimiçi 🟢")

# 6. ANA GÖVDE VE ANALİZ AKIŞI
st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Analiz için görüntü yükleyin", type=["jpg", "png", "tif"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file and model_service:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Kaynak Görüntü", use_container_width=True)

    with col2:
        # --- ANALİZ TETİKLEME ---
        if st.button("Analizi Başlat"):
            with st.spinner('Yapay zeka hücreleri tarıyor...'):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                results = model_service.predict(img_byte_arr.getvalue(), model_type=model_type)

                if results:
                    # Verileri Session State'e güvenle alıyoruz
                    st.session_state['results'] = results
                    st.session_state['analysis_done'] = True
                    st.session_state['last_count'] = results.get("detected_cells", 0)

        # --- SONUÇLARIN GÖSTERİLMESİ ---
        # .get() kullanarak KeyError (last_count bulunamadı) hatasını engelledik
        if st.session_state.get('analysis_done', False):
            res = st.session_state.get('results')
            count = st.session_state.get('last_count', 0)

            st.success(f"Analiz Tamamlandı!")
            st.metric("Tespit Edilen Hücre Sayısı", count)

            # --- OPSİYONEL E-POSTA RAPORLAMA ---
            st.markdown("---")
            st.subheader("📬 Analiz Raporunu Arşivle")

            email_input = st.text_input("E-posta adresi girin:", key="email_box", placeholder="doktor@hastane.com")

            if st.button("Raporu E-posta ile Gönder"):
                if email_input:
                    with st.spinner('Rapor iletiliyor...'):
                        success = send_pathovision_report(email_input, model_type, count)
                        if success:
                            st.balloons()
                            st.success(f"Başarılı! Rapor {email_input} adresine gönderildi.")
                        else:
                            st.error("E-posta gönderilemedi. SMTP ayarlarını kontrol edin.")
                else:
                    st.warning("Lütfen e-posta adresini boş bırakmayın.")

            with st.expander("Teknik Detayları Gör (JSON)"):
                st.json(res)