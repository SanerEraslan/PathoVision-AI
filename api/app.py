import streamlit as st
import io
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps
from fpdf import FPDF
import tempfile
import os

# --- PDF OLUŞTURMA FONKSİYONU (Görsel ve Grafik Destekli) ---
def create_pdf(results_data, model_type, original_img, chart_fig):
    pdf = FPDF()
    pdf.add_page()
    
    # Başlık
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(190, 20, "PATHOVISION AI - ANALIZ RAPORU", ln=True, align='C')
    
    # Rapor Bilgileri
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100)
    pdf.cell(190, 10, f"Rapor ID: PV-{pd.Timestamp.now().strftime('%Y%m%d%H%M')}", ln=True, align='R')
    pdf.cell(190, 5, f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
    pdf.ln(10)

    # 1. Sayısal Analiz Tablosu
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0)
    pdf.cell(190, 10, "1. Sayisal Analiz Sonuclari", ln=True)
    pdf.set_font("Arial", "", 11)
    
    for key, value in results_data.items():
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(90, 10, f" {key}", border=1, fill=True)
        pdf.cell(100, 10, f" {value}", border=1, ln=True)

    # 2. Görsel Analiz Bölümü
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "2. Mikroskobik ve Grafik Analiz", ln=True)
    
    current_y = pdf.get_y() + 5
    
    # Görseli geçici dosyaya kaydet ve PDF'e ekle
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        original_img.save(tmp_img.name)
        pdf.image(tmp_img.name, x=10, y=current_y, w=90)
    
    # Grafiği (Plotly) Byte olarak al ve PDF'e ekle
    # NOT: kaleido yüklü olmalıdır. Hata devam ederse 'pip install kaleido==0.2.1.post1' kullanın.
    try:
        img_bytes = chart_fig.to_image(format="png", engine="kaleido")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
            tmp_chart.write(img_bytes)
            tmp_chart.flush()
            pdf.image(tmp_chart.name, x=105, y=current_y, w=90)
    except Exception as e:
        pdf.set_xy(105, current_y)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(90, 10, "[Grafik yuklenemedi]", border=0)

    pdf.ln(75) # Görseller için dinamik boşluk
    
    # Yasal Uyarı Paneli
    pdf.set_y(-40)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150)
    pdf.multi_cell(190, 5, "UYARI: Bu belge yapay zeka tarafindan desteklenmis bir on analiz raporudur. Teshis degeri tasimaz. Lutfen uzman bir patolog tarafindan onaylanmis resmi raporu bekleyiniz.", align='C')
    
    # Bellek üzerinden PDF döndür
    return pdf.output(dest='S').encode('latin-1')

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-card { 
        background-color: white; 
        padding: 1.5rem; 
        border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063205.png", width=80)
    st.title("Ayarlar")
    uploaded_file = st.file_uploader("Doku Görüntüsü Yükle", type=["jpg", "png", "tif"])
    model_choice = st.selectbox("Model Seçimi", ["UNET++", "ResNet-V2"])
    conf_level = st.slider("Hassasiyet Eşiği", 0.1, 1.0, 0.7)

# --- ANA EKRAN ---
st.title("🔬 PathoVision AI: Dijital Patoloji Paneli")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    col_orig, col_proc = st.columns(2)
    
    with col_orig:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("🖼️ Ham Görüntü")
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Analizi Çalıştır", use_container_width=True):
        with st.spinner("Hücreler sınıflandırılıyor..."):
            # MODEL ÇIKTISI (Simülasyon)
            import time; time.sleep(1.2)
            processed_img = ImageOps.colorize(ImageOps.grayscale(img), black="black", white="#ff4b4b")
            
            t_cells, c_cells, h_cells = 142, 38, 104
            
            with col_proc:
                st.markdown('<div class="main-card">', unsafe_allow_html=True)
                st.subheader("🎯 Tespit Edilen Alanlar")
                st.image(processed_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Metrik Paneli
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Toplam Hücre", t_cells)
            m2.metric("Kanserli", c_cells, delta="Malign", delta_color="inverse")
            m3.metric("Sağlıklı", h_cells)
            m4.metric("Risk Skoru", f"%{round((c_cells/t_cells)*100, 1)}")

            # Grafik ve Raporlama
            c_left, c_right = st.columns([1.2, 1])
            
            with c_left:
                df = pd.DataFrame({"Sınıf": ["Kanserli", "Sağlıklı"], "Sayı": [c_cells, h_cells]})
                fig = px.pie(df, values='Sayı', names='Sınıf', hole=0.5,
                             color_discrete_sequence=["#ff4b4b", "#22c55e"])
                fig.update_layout(title="Hücre Dağılım Analizi")
                st.plotly_chart(fig, use_container_width=True)

            with c_right:
                st.markdown("### 📄 Rapor Oluştur")
                results_pdf = {
                    "Model": model_choice,
                    "Toplam Hucre": t_cells,
                    "Malign Hucre": c_cells,
                    "Benign Hucre": h_cells,
                    "Hassasiyet": f"{conf_level}"
                }
                
                # PDF Oluşturma (Hata Kontrollü)
                try:
                    pdf_output = create_pdf(results_pdf, model_choice, img, fig)
                    st.download_button(
                        label="📥 Analiz Raporunu PDF İndir",
                        data=pdf_output,
                        file_name="PathoVision_Rapor.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF oluşturulurken bir hata oluştu: {e}")
else:
    st.info("Analize başlamak için lütfen sol menüden bir mikroskop görüntüsü yükleyin.")
