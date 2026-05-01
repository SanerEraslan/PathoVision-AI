import streamlit as st
import io
import pandas as pd
import plotly.express as px
from PIL import Image
from fpdf import FPDF # PDF için

# --- PDF OLUŞTURMA FONKSİYONU ---
def create_pdf(results_data, model_type):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(190, 20, "PATHOVISION AI - ANALIZ RAPORU", ln=True, align='C')
    
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(100, 10, f"Analiz Tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.ln(7)
    pdf.cell(100, 10, f"Kullanilan Model: {model_type.upper()}")
    pdf.ln(15)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "ANALIZ SONUCLARI", ln=True)
    pdf.set_font("Arial", "", 12)
    
    for key, value in results_data.items():
        pdf.cell(90, 10, f"{key}:", border=1)
        pdf.cell(100, 10, f"{value}", border=1, ln=True)
        
    pdf.ln(20)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(190, 10, "Not: Bu rapor yapay zeka tarafindan olusturulmustur. Kesin teshis icin uzman doktor onayi gereklidir.")
    
    return pdf.output()

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="PathoVision AI", page_icon="🔬", layout="wide")

# Şık CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .report-card { background-color: white; padding: 30px; border-radius: 20px; border: 1px solid #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 PathoVision AI Laboratuvar Paneli")
st.write("Gelişmiş Histopatolojik Hücre Analiz Sistemi")

uploaded_file = st.file_uploader("Görüntü Yükle", type=["jpg", "png", "tif"])

if uploaded_file:
    # Modelin çalıştığını varsayalım
    if st.button("Analiz Et"):
        with st.spinner("Analiz ediliyor..."):
            # ÖRNEK VERİLER (Senin modelinden gelenlerle değişecek)
            total_cells = 150 
            cancer_cells = 45
            healthy_cells = 85
            background_ratio = 20 # Arka plan oranı (Yüzdesel)
            
            # --- SONUÇ EKRANI (PROFESYONEL TASARIM) ---
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            
            col_metrics, col_chart = st.columns([1, 1.5])
            
            with col_metrics:
                st.subheader("📋 Tespit Özetleri")
                st.metric("Toplam Hücre", total_cells)
                st.metric("Kanserli Hücre", f"{cancer_cells}", delta="Kritik", delta_color="inverse")
                st.metric("Sağlıklı Hücre", healthy_cells)
                st.write(f"**Model:** UNET++")
                st.write(f"**Doğruluk:** %92.4")

            with col_chart:
                # DAİRESEL GRAFİK (DONUT CHART)
                df_chart = pd.DataFrame({
                    "Kategori": ["Kanserli", "Sağlıklı", "Arka Plan"],
                    "Değer": [cancer_cells, healthy_cells, background_ratio]
                })
                
                fig = px.pie(df_chart, values='Değer', names='Kategori', hole=0.5,
                             color_discrete_sequence=["#ff4b4b", "#00cc96", "#636efa"],
                             title="Doku Dağılım Analizi")
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # --- RAPOR İNDİRME ---
            st.divider()
            st.subheader("💾 Resmi Raporu Oluştur")
            
            # PDF Verisi Hazırla
            results_for_pdf = {
                "Toplam Hucre Sayisi": total_cells,
                "Kanserli Hucre Sayisi": cancer_cells,
                "Saglikli Hucre Sayisi": healthy_cells,
                "Arka Plan Orani": f"%{background_ratio}",
                "Analiz Durumu": "Tamamlandi"
            }
            
            pdf_bytes = create_pdf(results_for_pdf, "unetplusplus")
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Analiz Raporunu PDF Olarak İndir",
                    data=bytes(pdf_bytes),
                    file_name="PathoVision_Rapor.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with col_dl2:
                # CSV alternatifi her zaman iyidir
                csv = df_chart.to_csv().encode('utf-8')
                st.download_button(
                    label="📊 Verileri Excel (CSV) Olarak İndir",
                    data=csv,
                    file_name="PathoVision_Veri.csv",
                    mime="text/csv",
                    use_container_width=True
                )
