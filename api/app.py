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

# --- MODEL YÜKLEME VE ÖNBELLEKLEME ---
@st.cache_resource
def load_patho_model(model_choice):
    # Kendi Hugging Face Kullanıcı Adını ve Repo İsmini Buraya Yaz
    REPO_ID = "SanerEraslan/PathoVision-Models" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seçilen modele göre mimariyi oluştur
    if "UNET++" in model_choice:
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        filename = "UNetPlusPlus_best.pth"
    else:
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        filename = "UNet_best.pth"
        
    try:
        # Modeli Hugging Face Hub'dan indir
        checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device).eval()
        return model, device
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None, device

# --- GERÇEK ANALİZ (INFERENCE) FONKSİYONU ---
def perform_real_segmentation(image, model, device):
    orig_w, orig_h = image.size
    
    # 1. Ön İşleme (Modelin beklediği 256x256 boyutu)
    input_img = image.resize((256, 256))
    img_array = np.array(input_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. Tahmin (Inference)
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # 3. Maskeyi Orijinal Boyuta Geri Döndür
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize((orig_w, orig_h))
    mask_final = np.array(mask_resized) / 255.0
    
    # 4. İstatistik ve Arkaplan Ayırımı
    # Arkaplan için parlaklık eşiği (Geleneksel yöntemle birleştirildi)
    grayscale = np.mean(np.array(image), axis=2)
    bg_mask = grayscale > 220
    
    # Kanserli maske (AI'dan gelen ve arkaplan olmayan alan)
    cancer_mask = (mask_final > 0.5) & (~bg_mask)
    healthy_mask = (~cancer_mask) & (~bg_mask)
    
    # İstatistikler
    total = orig_w * orig_h
    stats = {
        "Arkaplan": round((np.sum(bg_mask) / total) * 100, 1),
        "Saglikli": round((np.sum(healthy_mask) / total) * 100, 1),
        "Kanserli": round((np.sum(cancer_mask) / total) * 100, 1)
    }

    # 5. Görselleştirme (Overlay)
    img_rgba = image.convert("RGBA")
    overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    
    overlay[bg_mask] = [189, 195, 199, 60]      # Gri
    overlay[healthy_mask] = [46, 204, 113, 130] # Yeşil
    overlay[cancer_mask] = [231, 76, 60, 170]   # Kırmızı
    
    mask_layer = Image.fromarray(overlay, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=1))
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
    pdf.set_text_color(44, 62, 80)
    pdf.cell(190, 20, tr_fix("PATHOVISION AI - ANALIZ RAPORU"), ln=True, align='C')
    
    pdf.set_font("Arial", "", 10)
    pdf.cell(190, 10, tr_fix(f"Model: {model_name} | Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"), ln=True, align='R')
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, tr_fix("1. Alan Dagilim Verileri"), ln=True)
    pdf.set_font("Arial", "", 11)
    for label, val in stats.items():
        pdf.set_fill_color(245, 245, 245)
        pdf.cell(90, 10, tr_fix(f"{label} Alani"), border=1, fill=True)
        pdf.cell(100, 10, f" %{val}", border=1, ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, tr_fix("2. Segmentasyon Sonucu"), ln=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t_img:
        processed_img.save(t_img.name)
        pdf.image(t_img.name, x=10, y=pdf.get_y() + 5, w=170)
        t_img_path = t_img.name

    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 8)
    pdf.multi_cell(190, 5, tr_fix("UYARI: Bu belge yapay zeka tarafindan olusturulmustur. Tibbi teshis niteligi tasimaz."), align='C')
    
    os.unlink(t_img_path)
    return bytes(pdf.output(dest='S'))

# --- ANA EKRAN AYARLARI ---
st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063205.png", width=80)
    st.title("Kontrol Paneli")
    uploaded_file = st.file_uploader("Doku Görseli Yükle", type=["jpg", "png", "tif"])
    # MODEL İSİMLERİ GÜNCELLENDİ
    model_choice = st.selectbox("Model Seçimi", ["UNET++ (Nested)", "UNET (Standard)"])
    
    # Modeli otomatik yükle
    active_model, device = load_patho_model(model_choice)
    st.divider()
    st.info(f"Aktif Cihaz: {device.upper()}")

st.title("🔬 PathoVision AI: Dijital Patoloji Paneli")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    if st.button("🚀 AI Analizini Başlat", use_container_width=True):
        if active_model is not None:
            with st.spinner("Yapay zeka dokuyu hücre seviyesinde analiz ediyor..."):
                proc_img, stats = perform_real_segmentation(img, active_model, device)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🖼️ Ham Görüntü")
                    st.image(img, use_container_width=True)
                with c2:
                    st.subheader("🎯 AI Segmentasyon Haritası")
                    st.image(proc_img, use_container_width=True)

                st.divider()
                
                col_m, col_g = st.columns([1, 1.2])
                with col_m:
                    st.markdown("### 📊 Bölge Metrikleri")
                    st.metric("Kanserli Alan", f"%{stats['Kanserli']}", delta="Kritik", delta_color="inverse")
                    st.metric("Sağlıklı Alan", f"%{stats['Saglikli']}")
                    st.metric("Arkaplan", f"%{stats['Arkaplan']}")
                    
                    pdf_bytes = create_pdf(model_choice, proc_img, stats)
                    st.download_button("📥 PDF Analiz Raporunu İndir", pdf_bytes, "PathoVision_Rapor.pdf", "application/pdf", use_container_width=True)

                with col_g:
                    fig = px.pie(
                        values=[stats["Arkaplan"], stats["Saglikli"], stats["Kanserli"]], 
                        names=["Arkaplan", "Sağlıklı", "Kanserli"],
                        color=["Arkaplan", "Sağlıklı", "Kanserli"],
                        color_discrete_map={"Kanserli": "#e74c3c", "Sağlıklı": "#2ecc71", "Arkaplan": "#bdc3c7"},
                        hole=0.4
                    )
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Model yüklenemedi. Lütfen internet bağlantınızı veya Hugging Face repo ayarlarınızı kontrol edin.")
else:
    st.info("Lütfen sol menüden bir patoloji görüntüsü yükleyerek süreci başlatın.")
