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
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

# --- MODEL YÜKLEME VE ÖNBELLEKLEME (Hugging Face Entegrasyonu) ---
@st.cache_resource
def load_patho_model(model_choice):
    REPO_ID = "SanerEraslan/PathoVision-Models" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Teknik Karar: Modeller Binary (classes=1) olarak eğitilmiştir.
    if "UNET++" in model_choice:
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        filename = "UNetPlusPlus_best.pth"
    else:
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        filename = "UNet_best.pth"
        
    try:
        checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device).eval()
        return model, device
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None, device

# --- GERÇEK ANALİZ (INFERENCE) FONKSİYONU ---
def perform_real_segmentation(image, model, device):
    orig_w, orig_h = image.size
    
    # 1. Ön İşleme: Modelin beklediği 256x256 boyutu ve normalizasyon
    input_img = image.resize((256, 256))
    img_array = np.array(input_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. AI Tahmini (Sigmoid Aktivasyonu)
    with torch.no_grad():
        output = model(img_tensor)
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # 3. Maskeyi Orijinal Boyuta Çıkarma
    mask_resized = np.array(Image.fromarray((prob_mask * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)) / 255.0

    # 4. Arkaplan Ayıklama (Parlaklık Eşiği)
    grayscale = np.mean(np.array(image), axis=2)
    bg_mask = grayscale > 230 
    
    # 5. Sınıflandırma Mantığı
    final_cancer = (mask_resized > 0.5) & (~bg_mask)
    final_healthy = (~final_cancer) & (~bg_mask)
    final_bg = bg_mask

    # 6. İstatistiksel Hesaplama (Kantitatif Veri)
    total = orig_w * orig_h
    stats = {
        "Arkaplan": round((np.sum(final_bg) / total) * 100, 1),
        "Saglikli": round((np.sum(final_healthy) / total) * 100, 1),
        "Kanserli": round((np.sum(final_cancer) / total) * 100, 1)
    }

    # 7. İnovatif Görselleştirme (Alpha Blending)
    img_rgba = image.convert("RGBA")
    overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    
    overlay[final_bg] = [189, 195, 199, 60]      # Gri
    overlay[final_healthy] = [46, 204, 113, 130] # Yeşil
    overlay[final_cancer] = [231, 76, 60, 170]   # Kırmızı
    
    mask_layer = Image.fromarray(overlay, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=0.5))
    final_img = Image.alpha_composite(img_rgba, mask_layer).convert("RGB")
    
    return final_img, stats

# --- YARDIMCI FONKSİYONLAR ---
def tr_fix(text):
    chars = {"ğ": "g", "Ğ": "G", "ı": "i", "İ": "I", "ş": "s", "Ş": "S", 
             "ç": "c", "Ç": "C", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U"}
    for tr, lat in chars.items():
        text = str(text).replace(tr, lat)
    return text

def create_pdf(model_name, processed_img, stats):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(190, 20, tr_fix("PATHOVISION AI - ANALIZ RAPORU"), ln=True, align='C')
    
    pdf.set_font("Arial", "", 10)
    pdf.cell(190, 10, tr_fix(f"Model: {model_name} | Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"), ln=True, align='R')
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, tr_fix("1. Kantitatif Analiz Sonuclari"), ln=True)
    pdf.set_font("Arial", "", 12)
    for label, val in stats.items():
        pdf.cell(90, 10, tr_fix(f"{label} Alani:"), border=1)
        pdf.cell(100, 10, f" %{val}", border=1, ln=True)

    # Risk Notu Ekleme
    pdf.ln(5)
    if stats['Kanserli'] > 5:
        pdf.set_text_color(200, 0, 0)
        pdf.cell(190, 10, tr_fix("UYARI: Yuksek tumor tutulumu saptandi."), ln=True)
    pdf.set_text_color(0, 0, 0)

    # Görselleştirme Kaydı
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t_img:
        processed_img.save(t_img.name)
        pdf.image(t_img.name, x=10, y=100, w=190)
    
    return bytes(pdf.output(dest='S'))

# --- ARAYÜZ TASARIMI ---
st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063205.png", width=80)
    st.title("Kontrol Paneli")
    uploaded_file = st.file_uploader("Doku Görseli Yükle", type=["jpg", "png", "tif"])
    model_choice = st.selectbox("Model Seçimi", ["UNET++ Patho (Pro)", "UNET Patho (Fast)"])
    
    active_model, device = load_patho_model(model_choice)
    st.divider()
    st.write(f"**Donanım:** {device.upper()}")

st.title("🔬 PathoVision AI: Dijital Patoloji Paneli")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    if st.button("🚀 Analizi Başlat", use_container_width=True):
        if active_model:
            with st.spinner("Yapay Zeka dokuyu inceliyor..."):
                proc_img, stats = perform_real_segmentation(img, active_model, device)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🖼️ Ham Görüntü")
                    st.image(img, use_container_width=True)
                with c2:
                    st.subheader("🎯 Segmentasyon Haritası")
                    st.image(proc_img, use_container_width=True)

                st.divider()
                
                # --- SONUÇLAR VE RİSK SKALASI ---
                col_m, col_g, col_r = st.columns([1, 1, 1])
                
                with col_m:
                    st.markdown("### 📊 Bölge Metrikleri")
                    risk_color = "inverse" if stats['Kanserli'] > 1.0 else "normal"
                    st.metric("Kanserli Alan", f"%{stats['Kanserli']}", delta="RİSK", delta_color=risk_color)
                    st.metric("Sağlıklı Alan", f"%{stats['Saglikli']}")
                    
                    pdf_bytes = create_pdf(model_choice, proc_img, stats)
                    st.download_button("📥 PDF Raporu İndir", pdf_bytes, "Analiz_Raporu.pdf", use_container_width=True)

                with col_g:
                    fig = px.pie(
                        values=[stats["Arkaplan"], stats["Saglikli"], stats["Kanserli"]], 
                        names=["Arkaplan", "Sağlıklı", "Kanserli"],
                        color_discrete_map={"Kanserli": "#e74c3c", "Sağlıklı": "#2ecc71", "Arkaplan": "#bdc3c7"},
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_r:
                    st.markdown("### 🛑 Risk Cetveli")
                    st.markdown("""
                    | Oran | Seviye |
                    | :--- | :--- |
                    | %0-1 | <span style='color:#2ecc71'>Düşük</span> |
                    | %1-5 | <span style='color:#f39c12'>Orta</span> |
                    | %5-20| <span style='color:#e67e22'>Yüksek</span> |
                    | %20+ | <span style='color:#e74c3c'>Kritik</span> |
                    """, unsafe_allow_html=True)
        else:
            st.error("Model yüklenemedi!")
else:
    st.info("Analize başlamak için lütfen sol taraftan bir görsel yükleyin.")
