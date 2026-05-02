import streamlit as st
import io
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps
from fpdf import FPDF
import tempfile
import os

# --- PDF OLUŞTURMA FONKSİYONU (Görsel Destekli) ---
def create_pdf(results_data, model_type, original_img, chart_fig):
    pdf = FPDF()
    pdf.add_page()
    
    # Font Ayarı (Standart FPDF fontları Türkçe karakterde bazen sorun çıkarabilir)
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(190, 20, "PATHOVISION AI - ANALIZ RAPORU", ln=True, align='C')
    
    # Üst Bilgiler
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100)
    pdf.cell(190, 10, f"Rapor ID: PV-{pd.Timestamp.now().strftime('%Y%m%d%H%M')}", ln=True, align='R')
    pdf.cell(190, 5, f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
    pdf.ln(10)

    # Analiz Sonuçları Tablosu
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0)
    pdf.cell(190, 10, "1. Sayisal Analiz Sonuclari", ln=True)
    pdf.set_font("Arial", "", 11)
    
    for key, value in results_data.items():
        pdf.set_fill_color(245, 245, 245)
        pdf.cell(90, 10, f" {key}", border=1, fill=True)
        pdf.cell(100, 10, f" {value}", border=1, ln=True)

    # Görsel Analiz Bölümü
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "2. Mikroskobik Goruntuleme", ln=True)
    
    # Geçici dosyaya resmi kaydet ve PDF'e ekle
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        original_img.save(tmp_img.name)
        pdf.image(tmp_img.name, x=10, y=pdf.get_y() + 5, w=90)
    
    # Grafiği PDF'e ekle
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
        chart_fig.write_image(tmp_chart.name)
        pdf.image(tmp_chart.name, x=105, y=pdf.get_y() + 5, w=90)
    
    pdf.ln(70) # Görseller için boşluk
    
    # Yasal Uyarı
    pdf.set_y(-40)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150)
    pdf.multi_cell(190, 5, "UYARI: Bu belge yapay zeka tarafindan desteklenmis bir on analiz raporudur. Teshis degeri tasimaz. Lutfen uzman bir patolog tarafindan onaylanmis resmi raporu bekleyiniz.", align='C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

# Şık CSS Güncellemesi
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #1e3a8a; }
    .main-card { 
        background-color: white; 
        padding: 2rem; 
        border-radius: 15px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063205.png", width=100)
    st.title("Kontrol Paneli")
    uploaded_file = st.file_uploader("Görüntü Seçiniz", type=["jpg", "png", "tif"])
    st.info("Desteklenen formatlar: JPG, PNG, TIF. Maksimum dosya boyutu: 10MB")
    
    model_choice = st.selectbox("Analiz Modeli", ["UNET++ (Yüksek Hassasiyet)", "ResNet-50 (Hızlı Tarama)"])
    confidence_threshold = st.slider("Güven Eşiği", 0.0, 1.0, 0.5)

# --- ANA EKRAN ---
st.title("🔬 PathoVision AI: Hücre Analiz Sistemi")

if uploaded_file:
    # Görüntüyü Hazırla
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("🖼️ Orijinal Görüntü")
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Analiz Butonu
    if st.button("🚀 Analizi Başlat", use_container_width=True):
        with st.spinner("Yapay zeka dokuları ayrıştırıyor..."):
            # --- MODEL SİMÜLASYONU (Burası senin model çıktılarınla değişecek) ---
            import time; time.sleep(1) # Gerçekçi bir bekleme
            
            # Örnek maske oluşturma (Simülasyon için resmi ters çeviriyoruz)
            processed_img = ImageOps.colorize(ImageOps.grayscale(img), black="black", white="red")
            
            total_cells = 142
            cancer_cells = 38
            healthy_cells = 104
            
            with col2:
                st.markdown('<div class="main-card">', unsafe_allow_html=True)
                st.subheader("🎯 Analiz Edilmiş Görüntü")
                st.image(processed_img, use_container_width=True, caption="Kırmızı alanlar potansiyel malign hücreleri temsil eder.")
                st.markdown('</div>', unsafe_allow_html=True)

            # --- İSTATİSTİKLER ---
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Toplam Hücre", total_cells)
            m2.metric("Kanserli Hücre", cancer_cells, delta="Kritik", delta_color="inverse")
            m3.metric("Sağlıklı Hücre", healthy_cells)
            m4.metric("Malignite Oranı", f"%{round((cancer_cells/total_cells)*100, 1)}")

            # --- GRAFİKLER ---
            c1, c2 = st.columns([1.5, 1])
            
            with c1:
                df_chart = pd.DataFrame({
                    "Sınıf": ["Kanserli", "Sağlıklı"],
                    "Sayı": [cancer_cells, healthy_cells]
                })
                fig = px.pie(df_chart, values='Sayı', names='Sınıf', hole=0.6,
                             color_discrete_sequence=["#ef4444", "#10b981"],
                             title="Hücre Dağılım Oranı")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.write("### 📝 Analiz Notları")
                st.write(f"- **Kullanılan Model:** {model_choice}")
                st.write(f"- **Tespit Hassasiyeti:** %94.2")
                st.write("- **Öneri:** Görüntüde yoğun kümelenme gözlemlendi. İleri tetkik önerilir.")

            # --- PDF OLUŞTURMA ---
            results_for_pdf = {
                "Toplam Hucre": total_cells,
                "Kanserli Hucre": cancer_cells,
                "Saglikli Hucre": healthy_cells,
                "Model": model_choice,
                "Guven Esigi": confidence_threshold
            }
            
            # Grafik dosyasını PDF için hazırla
            pdf_data = create_pdf(results_for_pdf, "unet", img, fig)
            
            st.download_button(
                label="📄 Profesyonel Analiz Raporunu İndir (PDF)",
                data=pdf_data,
                file_name="PathoVision_Analiz_Raporu.pdf",
                mime="application/pdf",
                use_container_width=True
            )
else:
    st.warning("Lütfen analiz için bir mikroskop görüntüsü yükleyin.")
