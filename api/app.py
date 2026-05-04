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
    # Kendi Hugging Face Bilgilerin
    REPO_ID = "SanerEraslan/PathoVision-Models" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # HATA ÇÖZÜMÜ: classes=3 olarak ayarlandı (Model ağırlıklarınız 3 sınıf içeriyor)
    if "UNET++" in model_choice:
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
        filename = "evrensel_kanser_modeli_pro.pth"
    else:
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
        filename = "evrensel_kanser_modeli.pth"
        
    try:
        checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        # map_location cihaz uyumsuzluklarını önler
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device).eval()
        return model, device
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None, device

# --- GERÇEK ANALİZ (INFERENCE) FONKSİYONU ---
def perform_real_segmentation(image, model, device):
    orig_w, orig_h = image.size
    
    # 1. Ön İşleme
    input_img = image.resize((256, 256))
    img_array = np.array(input_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. Tahmin (Multi-class Inference)
    with torch.no_grad():
        output = model(img_tensor)
        # Softmax ile olasılıkları 3 kanala dağıtıyoruz
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    
    # 3. Kanalları Ayırma ve Boyutlandırma
    # probs[0] = Arkaplan, probs[1] = Sağlıklı, probs[2] = Kanserli
    def resize_mask(mask_array):
        return np.array(Image.fromarray((mask_array * 255).astype(np.uint8)).resize((orig_w, orig_h))) / 255.0

    bg_ai = resize_mask(probs[0])
    healthy_ai = resize_mask(probs[1])
    cancer_ai = resize_mask(probs[2])

    # 4. Final Karar (En yüksek olasılığa sahip sınıfı seçme)
    # AI bazen boş alanları da doku sanabilir, parlaklık eşiğiyle destekliyoruz
    grayscale = np.mean(np.array(image), axis=2)
    hard_bg_mask = grayscale > 235 # Çok parlak alanlar kesin arkaplandır
    
    final_cancer = (cancer_ai > healthy_ai) & (cancer_ai > bg_ai) & (~hard_bg_mask)
    final_healthy = (healthy_ai >= cancer_ai) & (healthy_ai > bg_ai) & (~hard_bg_mask)
    final_bg = ~(final_cancer | final_healthy)

    # 5. İstatistikler
    total = orig_w * orig_h
    stats = {
        "Arkaplan": round((np.sum(final_bg) / total) * 100, 1),
        "Saglikli": round((np.sum(final_healthy) / total) * 100, 1),
        "Kanserli": round((np.sum(final_cancer) / total) * 100, 1)
    }

    # 6. Görselleştirme (Overlay)
    img_rgba = image.convert("RGBA")
    overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    
    overlay[final_bg] = [189, 195, 199, 60]      # Gri
    overlay[final_healthy] = [46, 204, 113, 130] # Yeşil
    overlay[final_cancer] = [231, 76, 60, 170]   # Kırmızı
    
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
    pdf.cell(190, 10, tr_fix("2. Segmentasyon Haritasi ve Grafik"), ln=True)
    
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
    
    active_model, device = load_patho_model(model_choice)
    st.divider()
    st.info(f"Aktif Cihaz: {device.upper()}")

st.title("🔬 PathoVision AI: Dijital Patoloji Paneli")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    if st.button("🚀 Analizi Çalıştır", use_container_width=True):
        if active_model is not None:
            with st.spinner("AI dokuyu analiz ediyor..."):
                proc_img, stats = perform_real_segmentation(img, active_model, device)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🖼️ Ham Görüntü")
                    st.image(img, use_container_width=True)
                with c2:
                    st.subheader("🎯 Segmentasyon Haritası")
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
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Model yüklenemedi, lütfen tekrar deneyin.")
else:
    st.info("Lütfen sol menüden bir patoloji görüntüsü yükleyerek süreci başlatın.")
