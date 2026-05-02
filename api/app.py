import streamlit as st
import io
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps, ImageDraw
from fpdf import FPDF
import tempfile
import os

# --- YARDIMCI FONKSİYONLAR ---
def tr_fix(text):
    """PDF için Türkçe karakterleri standart Latin karakterlerine dönüştürür."""
    chars = {"ğ": "g", "Ğ": "G", "ı": "i", "İ": "I", "ş": "s", "Ş": "S", 
             "ç": "c", "Ç": "C", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U"}
    for tr, lat in chars.items():
        text = str(text).replace(tr, lat)
    return text

def perform_analysis_sim(image):
    """Resmin tamamını boyamak yerine üzerine şeffaf hücre işaretleri ekler."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    
    # Kanserli hücreler (Kırmızı)
    for _ in range(15):
        import random
        x, y = random.randint(int(w*0.2), int(w*0.5)), random.randint(int(h*0.2), int(h*0.5))
        draw.ellipse([x-15, y-15, x+15, y+15], fill=(255, 0, 0, 100), outline=(255, 0, 0, 200))
        
    # Sağlıklı hücreler (Yeşil)
    for _ in range(25):
        import random
        x, y = random.randint(int(w*0.5), int(w*0.8)), random.randint(int(h*0.4), int(h*0.8))
        draw.ellipse([x-12, y-12, x+12, y+12], fill=(0, 255, 0, 100), outline=(0, 255, 0, 200))
        
    return Image.alpha_composite(base, overlay).convert("RGB")

# --- PDF OLUŞTURMA FONKSİYONU ---
def create_pdf(results_data, model_type, processed_img, chart_fig):
    pdf = FPDF()
    pdf.add_page()
    
    # Başlık
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(190, 20, tr_fix("PATHOVISION AI - ANALIZ RAPORU"), ln=True, align='C')
    
    # Rapor Bilgileri
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100)
    pdf.cell(190, 10, tr_fix(f"Rapor ID: PV-{pd.Timestamp.now().strftime('%Y%m%d%H%M')}"), ln=True, align='R')
    pdf.cell(190, 5, tr_fix(f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"), ln=True, align='R')
    pdf.ln(10)

    # 1. Sayısal Analiz Tablosu
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0)
    pdf.cell(190, 10, tr_fix("1. Sayisal Analiz Sonuclari"), ln=True)
    pdf.set_font("Arial", "", 11)
    
    for key, value in results_data.items():
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(90, 10, f" {tr_fix(key)}", border=1, fill=True)
        pdf.cell(100, 10, f" {tr_fix(value)}", border=1, ln=True)

    # 2. Görsel Analiz Bölümü
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, tr_fix("2. Mikroskobik ve Grafik Analiz"), ln=True)
    
    current_y = pdf.get_y() + 5
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        processed_img.save(tmp_img.name)
        pdf.image(tmp_img.name, x=10, y=current_y, w=90)
    
    try:
        img_bytes = chart_fig.to_image(format="png", engine="kaleido")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
            tmp_chart.write(img_bytes)
            tmp_chart.flush()
            pdf.image(tmp_chart.name, x=105, y=current_y, w=90)
    except:
        pdf.set_xy(105, current_y)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(90, 10, "[Grafik yuklenemedi]", border=0)

    pdf.set_y(-40)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150)
    warning_text = "UYARI: Bu belge yapay zeka destekli bir on analiz raporudur. Teshis degeri tasimaz."
    pdf.multi_cell(190, 5, tr_fix(warning_text), align='C')
    
    # --- KRİTİK DÜZELTME: BYTEARRAY HATASI İÇİN ---
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return bytes(pdf_output) # bytearray gelirse saf bytes'a çevirir

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

# CSS Kart Tasarımı
st.markdown("""
    <style>
    .main-card { background-color: white; padding: 1.5rem; border-radius: 12px; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 20px; border: 1px solid #eee; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063205.png", width=80)
    st.title("Kontrol Paneli")
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
        with st.spinner("Hücreler analiz ediliyor..."):
            processed_img = perform_analysis_sim(img)
            t_cells, c_cells, h_cells = 142, 38, 104
            
            with col_proc:
                st.markdown('<div class="main-card">', unsafe_allow_html=True)
                st.subheader("🎯 Analiz Sonucu")
                st.image(processed_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Toplam Hücre", t_cells)
            m2.metric("Kanserli", c_cells, delta="Malign", delta_color="inverse")
            m3.metric("Sağlıklı", h_cells)
            m4.metric("Risk Skoru", f"%{round((c_cells/t_cells)*100, 1)}")

            c_left, c_right = st.columns([1.2, 1])
            with c_left:
                df = pd.DataFrame({"Sinif": ["Kanserli", "Saglikli"], "Sayi": [c_cells, h_cells]})
                fig = px.pie(df, values='Sayi', names='Sinif', hole=0.5, color_discrete_sequence=["#ff4b4b", "#22c55e"])
                st.plotly_chart(fig, use_container_width=True)

            with c_right:
                st.markdown("### 📄 Rapor Oluştur")
                results_pdf_data = {
                    "Model Tipi": model_choice,
                    "Toplam Hucre Sayisi": t_cells,
                    "Malign Hucre": c_cells,
                    "Benign Hucre": h_cells,
                    "Analiz Hassasiyeti": f"{conf_level}"
                }
                
                try:
                    pdf_data = create_pdf(results_pdf_data, model_choice, processed_img, fig)
                    st.download_button(
                        label="📥 Analiz Raporunu PDF İndir",
                        data=pdf_data,
                        file_name="PathoVision_Analiz_Raporu.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Rapor hatası: {e}")
else:
    st.info("Lütfen bir mikroskop görüntüsü yükleyerek analizi başlatın.")
