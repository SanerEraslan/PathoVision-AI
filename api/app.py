import streamlit as st
import io
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps, ImageDraw
from fpdf import FPDF
import tempfile
import os

# --- PDF OLUŞTURMA FONKSİYONU ---
def create_pdf(results_data, model_type, processed_img, chart_fig):
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
    
    # Analiz edilmiş görseli PDF'e ekle
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        processed_img.save(tmp_img.name)
        pdf.image(tmp_img.name, x=10, y=current_y, w=90)
    
    # Grafiği PDF'e ekle
    try:
        img_bytes = chart_fig.to_image(format="png", engine="kaleido")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
            tmp_chart.write(img_bytes)
            tmp_chart.flush()
            pdf.image(tmp_chart.name, x=105, y=current_y, w=90)
    except:
        pdf.set_xy(105, current_y)
        pdf.cell(90, 10, "[Grafik Hatasi]", border=0)

    pdf.set_y(-40)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150)
    pdf.multi_cell(190, 5, "UYARI: Bu rapor yapay zeka yardimiyla hazirlanmistir. Kesin teshis icin doktor onayi sarttir.", align='C')
    
    # HATA DÜZELTME: bytearray/bytes kontrolü
    output = pdf.output(dest='S')
    if isinstance(output, str):
        return output.encode('latin-1')
    return output

# --- ANALİZ SİMÜLASYONU (Kırmızı ve Yeşil İşaretleme) ---
def perform_analysis_sim(image):
    # Orijinal resmi taban olarak al
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    w, h = base.size
    # Kanserli hücre simülasyonu (Kırmızı)
    for _ in range(12):
        x, y = w * 0.4, h * 0.3
        draw.ellipse([x-30, y-30, x+30, y+30], fill=(255, 0, 0, 80), outline=(255, 0, 0, 200))
        
    # Sağlıklı hücre simülasyonu (Yeşil)
    for _ in range(20):
        x, y = w * 0.6, h * 0.7
        draw.ellipse([x-25, y-25, x+25, y+25], fill=(0, 255, 0, 80), outline=(0, 255, 0, 200))
        
    return Image.alpha_composite(base, overlay).convert("RGB")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

# --- ANA EKRAN ---
st.title("🔬 PathoVision AI: Hücre Analiz Paneli")

if uploaded_file := st.sidebar.file_uploader("Doku Görüntüsü", type=["jpg", "png"]):
    img = Image.open(uploaded_file).convert("RGB")
    
    if st.button("🚀 Analizi Başlat", use_container_width=True):
        processed_img = perform_analysis_sim(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Ham Görüntü", use_container_width=True)
        with col2:
            st.image(processed_img, caption="İşaretlenmiş Görüntü (Kırmızı: Malign, Yeşil: Benign)", use_container_width=True)
            
        # Sonuçlar
        t_cells, c_cells, h_cells = 150, 45, 105
        
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Toplam Hücre", t_cells)
        m2.metric("Kanserli", c_cells, delta="Malign", delta_color="inverse")
        m3.metric("Sağlıklı", h_cells)
        
        # Grafik
        df = pd.DataFrame({"Sınıf": ["Kanserli", "Sağlıklı"], "Sayı": [c_cells, h_cells]})
        fig = px.pie(df, values='Sayı', names='Sınıf', hole=0.5, color_discrete_sequence=["#ef4444", "#22c55e"])
        st.plotly_chart(fig, use_container_width=True)
        
        # PDF Raporu
        try:
            results_data = {"Toplam Hücre": t_cells, "Kanserli": c_cells, "Sağlıklı": h_cells}
            pdf_bytes = create_pdf(results_data, "UNET++", processed_img, fig)
            
            st.download_button(
                label="📥 Analiz Raporunu İndir",
                data=pdf_bytes,
                file_name="PathoVision_Analiz.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Rapor oluşturma hatası: {e}")
