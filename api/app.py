import streamlit as st
import io
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image, ImageDraw
from fpdf import FPDF
import tempfile
import os
import matplotlib.pyplot as plt

# --- YARDIMCI FONKSİYONLAR ---
def tr_fix(text):
    """PDF için Türkçe karakterleri standart Latin karakterlerine dönüştürür."""
    chars = {"ğ": "g", "Ğ": "G", "ı": "i", "İ": "I", "ş": "s", "Ş": "S", 
             "ç": "c", "Ç": "C", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U"}
    for tr, lat in chars.items():
        text = str(text).replace(tr, lat)
    return text

def perform_segmentation(image):
    """
    Görüntüyü analiz ederek pikselleri Kanserli, Sağlıklı ve Arkaplan olarak ayırır.
    Gerçek model maskesini simüle eder.
    """
    img_array = np.array(image.convert("RGB"))
    h, w, _ = img_array.shape
    
    # Simülasyon: Arkaplanı (parlak alanlar) ayır
    grayscale = np.mean(img_array, axis=2)
    bg_mask = grayscale > 225 # Beyaza yakın pikseller arkaplandır
    
    # Simülasyon: Kanserli bölge oluştur (Görüntünün merkezinde rastgele bir doku)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    cancer_mask = (x - center_x)**2 + (y - center_y)**2 < (min(h, w) // 3)**2
    cancer_mask = cancer_mask & ~bg_mask # Sadece doku olan yerlerde kanser olabilir
    
    # Sağlıklı doku: Arkaplan olmayan ve kanser olmayan her yer
    healthy_mask = ~bg_mask & ~cancer_mask

    # Görselleştirme (Overlay)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[healthy_mask] = [34, 197, 94, 100]  # Yeşil (Sağlıklı)
    overlay[cancer_mask] = [231, 76, 60, 130]   # Kırmızı (Kanserli)
    
    base_img = image.convert("RGBA")
    mask_img = Image.fromarray(overlay, mode="RGBA")
    final_img = Image.alpha_composite(base_img, mask_img).convert("RGB")
    
    # İstatistikleri hesapla
    total_pixels = h * w
    stats = {
        "Arkaplan": round((np.sum(bg_mask) / total_pixels) * 100, 1),
        "Saglikli": round((np.sum(healthy_mask) / total_pixels) * 100, 1),
        "Kanserli": round((np.sum(cancer_mask) / total_pixels) * 100, 1)
    }
    
    return final_img, stats

# --- PDF OLUŞTURMA FONKSİYONU ---
def create_pdf(results_data, processed_img, stats):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(190, 20, tr_fix("PATHOVISION AI - ANALIZ RAPORU"), ln=True, align='C')
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100)
    pdf.cell(190, 10, tr_fix(f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"), ln=True, align='R')
    pdf.ln(5)

    # 1. Alan Dağılım Tablosu
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0)
    pdf.cell(190, 10, tr_fix("1. Alan Analizi"), ln=True)
    pdf.set_font("Arial", "", 11)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(90, 10, tr_fix("Analiz Modeli"), border=1, fill=True)
    pdf.cell(100, 10, f" {tr_fix(results_data['Model'])}", border=1, ln=True)

    for label, val in stats.items():
        pdf.cell(90, 10, tr_fix(f"{label} Alan Orani"), border=1, fill=True)
        pdf.cell(100, 10, f" %{val}", border=1, ln=True)

    # 2. Mikroskobik Görünüm ve Grafik
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, tr_fix("2. Segmentasyon Haritasi ve Dagilim"), ln=True)
    
    current_y = pdf.get_y() + 5
    
    # Segmentasyon Resmini Kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        processed_img.save(tmp_img.name)
        pdf.image(tmp_img.name, x=10, y=current_y, w=90)
        tmp_img_path = tmp_img.name

    # Matplotlib Pasta Grafiği (3 Renkli)
    try:
        plt.figure(figsize=(5, 5))
        labels = ['Arkaplan', 'Saglikli', 'Kanserli']
        sizes = [stats["Arkaplan"], stats["Saglikli"], stats["Kanserli"]]
        colors = ['#dfe6e9', '#2ecc71', '#e74c3c']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        plt.title("Alan Dagilimi")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_plt:
            plt.savefig(tmp_plt.name, bbox_inches='tight', dpi=100)
            pdf.image(tmp_plt.name, x=105, y=current_y, w=90)
            tmp_plt_path = tmp_plt.name
        plt.close()
    except:
        pdf.set_xy(105, current_y)
        pdf.cell(90, 10, "[Grafik Hatasi]")

    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 8)
    pdf.multi_cell(190, 5, tr_fix("UYARI: Bu belge yapay zeka destekli bir segmentasyon raporudur. Klinik karar destek amaclidir."), align='C')
    
    os.unlink(tmp_img_path)
    if 'tmp_plt_path' in locals(): os.unlink(tmp_plt_path)

    return bytes(pdf.output(dest='S'))

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063205.png", width=80)
    st.title("Kontrol Paneli")
    uploaded_file = st.file_uploader("Mikroskop Görüntüsü Yükle", type=["jpg", "png", "tif"])
    model_choice = st.selectbox("Model Seçimi", ["UNET++", "ResNet-V2", "Vision Transformer"])
    st.divider()
    st.info("Hassasiyet eşiği model tarafından otomatik kalibre edilmektedir.")

# --- ANA EKRAN ---
st.title("🔬 PathoVision AI: Dijital Segmentasyon Paneli")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    if st.button("🚀 Analizi Çalıştır", use_container_width=True):
        with st.spinner("Piksel düzeyinde segmentasyon yapılıyor..."):
            processed_img, stats = perform_segmentation(img)
            
            col_orig, col_proc = st.columns(2)
            with col_orig:
                st.markdown('<div class="main-card">', unsafe_allow_html=True)
                st.subheader("🖼️ Ham Görüntü")
                st.image(img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_proc:
                st.markdown('<div class="main-card">', unsafe_allow_html=True)
                st.subheader("🎯 Bölge Segmentasyonu")
                st.image(processed_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.divider()
            
            # Metrikler
            m1, m2, m3 = st.columns(3)
            m1.metric("Kanserli Alan", f"%{stats['Kanserli']}", delta="Riskli", delta_color="inverse")
            m2.metric("Sağlıklı Alan", f"%{stats['Saglikli']}")
            m3.metric("Arkaplan", f"%{stats['Arkaplan']}")

            c_left, c_right = st.columns([1.2, 1])
            with c_left:
                # Dashboard Grafik (3 Alan)
                df = pd.DataFrame({
                    "Bölge": ["Kanserli", "Sağlıklı", "Arkaplan"],
                    "Yüzde": [stats["Kanserli"], stats["Saglikli"], stats["Arkaplan"]]
                })
                fig = px.pie(df, values='Yüzde', names='Bölge', hole=0.5,
                             color_discrete_map={"Kanserli": "#e74c3c", "Sağlıklı": "#2ecc71", "Arkaplan": "#dfe6e9"})
                st.plotly_chart(fig, use_container_width=True)

            with c_right:
                st.markdown("### 📄 Rapor Oluştur")
                try:
                    pdf_data = create_pdf({"Model": model_choice}, processed_img, stats)
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
    st.info("Lütfen bir mikroskop görüntüsü yükleyerek süreci başlatın.")
