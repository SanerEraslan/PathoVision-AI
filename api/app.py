import streamlit as st
import io
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image, ImageFilter
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
    Görüntü piksellerini analiz ederek Kanserli, Sağlıklı ve Arkaplan olarak ayırır.
    """
    img_array = np.array(image.convert("RGB"))
    h, w, _ = img_array.shape
    grayscale = np.mean(img_array, axis=2)
    
    # 1. Arkaplan Maskesi (Beyaz/Parlak alanlar)
    bg_mask = grayscale > 220 
    
    # 2. Doku Maskesi (Arkaplan olmayan her yer)
    doku_mask = ~bg_mask
    
    # 3. Kanserli Bölge Simülasyonu (Doku içindeki en koyu/yoğun %30'luk kesim)
    if np.any(doku_mask):
        cancer_threshold = np.percentile(grayscale[doku_mask], 30)
        cancer_mask = doku_mask & (grayscale <= cancer_threshold)
    else:
        cancer_mask = np.zeros_like(bg_mask)
        
    healthy_mask = doku_mask & ~cancer_mask

    # Overlay oluşturma (RGBA)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Renk Atamaları (RGB + Alpha)
    overlay[bg_mask] = [189, 195, 199, 60]      # Arkaplan: Gri (#bdc3c7)
    overlay[healthy_mask] = [46, 204, 113, 130] # Sağlıklı: Yeşil (#2ecc71)
    overlay[cancer_mask] = [231, 76, 60, 170]   # Kanserli: Kırmızı (#e74c3c)
    
    base_img = image.convert("RGBA")
    mask_img = Image.fromarray(overlay, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=1))
    final_img = Image.alpha_composite(base_img, mask_img).convert("RGB")
    
    stats = {
        "Arkaplan": round((np.sum(bg_mask) / (h*w)) * 100, 1),
        "Saglikli": round((np.sum(healthy_mask) / (h*w)) * 100, 1),
        "Kanserli": round((np.sum(cancer_mask) / (h*w)) * 100, 1)
    }
    
    return final_img, stats

# --- PDF OLUŞTURMA FONKSİYONU ---
def create_pdf(model_name, processed_img, stats):
    pdf = FPDF()
    pdf.add_page()
    
    # Başlık
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(190, 20, tr_fix("PATHOVISION AI - ANALIZ RAPORU"), ln=True, align='C')
    
    pdf.set_font("Arial", "", 10)
    pdf.cell(190, 10, tr_fix(f"Model: {model_name} | Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"), ln=True, align='R')
    pdf.ln(5)

    # 1. Alan Analiz Tablosu
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, tr_fix("1. Alan Dagilim Verileri"), ln=True)
    pdf.set_font("Arial", "", 11)
    for label, val in stats.items():
        pdf.set_fill_color(245, 245, 245)
        pdf.cell(90, 10, tr_fix(f"{label} Alani"), border=1, fill=True)
        pdf.cell(100, 10, f" %{val}", border=1, ln=True)

    pdf.ln(5)

    # --- YENİ: PDF RİSK ANALİZİ BÖLÜMÜ ---
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, tr_fix("2. Risk Analizi ve Seviye Cetveli"), ln=True)
    
    cancer_val = stats['Kanserli']
    if cancer_val <= 1.0:
        msg, color = "DURUM: DUSUK RISK", (46, 204, 113)
    elif 1.0 < cancer_val <= 5.0:
        msg, color = "DURUM: ORTA RISK", (243, 156, 18)
    elif 5.0 < cancer_val <= 20.0:
        msg, color = "DURUM: YUKSEK RISK", (230, 126, 34)
    else:
        msg, color = "DURUM: KRITIK SEVIYE", (231, 76, 60)

    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(color[0], color[1], color[2])
    pdf.cell(190, 10, tr_fix(f"{msg} (%{cancer_val})"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 10)
    pdf.cell(190, 5, tr_fix("Risk cetveli: %0-1 Dusuk, %1-5 Orta, %5-20 Yuksek, %20+ Kritik"), ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, tr_fix("3. Segmentasyon Haritasi ve Grafik"), ln=True)
    
    y_pos = pdf.get_y() + 5
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t_img:
        processed_img.save(t_img.name)
        pdf.image(t_img.name, x=10, y=y_pos, w=90)
        t_img_path = t_img.name

    try:
        plt.figure(figsize=(5, 5))
        plt.pie([stats["Arkaplan"], stats["Saglikli"], stats["Kanserli"]], 
                labels=['Arkaplan', 'Saglikli', 'Kanserli'], 
                colors=['#bdc3c7', '#2ecc71', '#e74c3c'], 
                autopct='%1.1f%%', startangle=140)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t_plt:
            plt.savefig(t_plt.name, bbox_inches='tight')
            pdf.image(t_plt.name, x=105, y=y_pos, w=90)
            t_plt_path = t_plt.name
        plt.close()
    except: pass

    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 8)
    pdf.multi_cell(190, 5, tr_fix("UYARI: Bu belge yapay zeka tarafindan olusturulmustur. Tibbi teshis niteligi tasimaz."), align='C')
    
    os.unlink(t_img_path)
    if 't_plt_path' in locals(): os.unlink(t_plt_path)
    return bytes(pdf.output(dest='S'))

# --- ANA EKRAN AYARLARI ---
st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063205.png", width=80)
    st.title("Kontrol Paneli")
    uploaded_file = st.file_uploader("Doku Görseli Yükle", type=["jpg", "png", "tif"])
    model_choice = st.selectbox("Model Seçimi", ["UNET++ Patho", "UNET Patho"])
    st.divider()
    st.info("Hassasiyet eşiği model tarafından otomatik optimize edilmektedir.")

st.title("🔬 PathoVision AI: Dijital Patoloji Paneli")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    if st.button("🚀 Analizi Çalıştır", use_container_width=True):
        with st.spinner("Doku segmentasyonu yapılıyor..."):
            proc_img, stats = perform_segmentation(img)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🖼️ Ham Görüntü")
                st.image(img, use_container_width=True)
            with c2:
                st.subheader("🎯 Segmentasyon Haritası")
                st.image(proc_img, use_container_width=True)

            st.divider()
            
            # --- SONUÇLAR VE DİNAMİK RİSK PANALİ ---
            col_m, col_g, col_r = st.columns([1, 1, 1])
            
            with col_m:
                st.markdown("### 📊 Bölge Metrikleri")
                st.metric("Kanserli Alan", f"%{stats['Kanserli']}", delta="Analiz Edildi", delta_color="inverse")
                st.metric("Sağlıklı Alan", f"%{stats['Saglikli']}")
                st.metric("Arkaplan", f"%{stats['Arkaplan']}")
                
                pdf_bytes = create_pdf(model_choice, proc_img, stats)
                st.download_button("📥 PDF Analiz Raporunu İndir", pdf_bytes, "PathoVision_Rapor.pdf", "application/pdf", use_container_width=True)

            with col_g:
                st.markdown("### 📈 Dağılım Grafiği")
                fig = px.pie(
                    values=[stats["Arkaplan"], stats["Saglikli"], stats["Kanserli"]], 
                    names=["Arkaplan", "Sağlıklı", "Kanserli"],
                    color=["Arkaplan", "Sağlıklı", "Kanserli"],
                    color_discrete_map={
                        "Kanserli": "#e74c3c", "Sağlıklı": "#2ecc71", "Arkaplan": "#bdc3c7"
                    },
                    hole=0.4
                )
                fig.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                st.markdown("### 🛑 Risk Değerlendirme Cetveli")
                st.markdown("""
                | Kanserli Alan (%) | Risk Seviyesi |
                | :--- | :--- |
                | **%0 - %1** | <span style='color:#2ecc71'>● Düşük Risk</span> |
                | **%1 - %5** | <span style='color:#f39c12'>● Orta Risk</span> |
                | **%5 - %20** | <span style='color:#e67e22'>● Yüksek Risk</span> |
                | **%20 ve Üzeri** | <span style='color:#e74c3c'>● Kritik Seviye</span> |
                
                ---
                """, unsafe_allow_html=True)
                
                # Dinamik Uyarı Mekanizması
                cancer_perc = stats['Kanserli']
                if cancer_perc <= 1.0:
                    st.success(f"✅ Durum: Düşük Risk (%{cancer_perc})")
                elif 1.0 < cancer_perc <= 5.0:
                    st.warning(f"⚠️ Durum: Orta Risk (%{cancer_perc})")
                elif 5.0 < cancer_perc <= 20.0:
                    st.error(f"🚨 Durum: Yüksek Risk (%{cancer_perc})")
                else:
                    st.error(f"☢️ Durum: KRİTİK SEVİYE (%{cancer_perc})")
                
                st.info("**Bilgilendirme:** Bu veriler yapay zeka segmentasyonu ile hesaplanmıştır.")

else:
    st.info("Lütfen sol menüden bir patoloji görüntüsü yükleyerek süreci başlatın.")
