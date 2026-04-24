import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_pathovision_report(target_email, model_name, cell_count):
    # --- OUTLOOK SMTP AYARLARI ---
    SMTP_SERVER = "smtp.office365.com"  # Outlook/Hotmail için standart sunucu
    SMTP_PORT = 587
    USERNAME = "phatovision@outlook.com"  # Buraya kendi adresini yaz
    PASSWORD = "7DA9A-YZZXT-WUBZ8-E93NK-4QDR6"  # Aldığın uygulama şifresini yaz

    message = MIMEMultipart()
    message["From"] = USERNAME
    message["To"] = target_email
    message["Subject"] = f"PathoVision AI Analiz Raporu"

    body = f"""
    <html>
      <body style="font-family: sans-serif; padding: 20px;">
        <h2 style="color: #2c3e50;">🔬 PathoVision AI Raporu</h2>
        <hr>
        <p><b>Kullanılan Model:</b> {model_name}</p>
        <p><b>Tespit Edilen Hücre:</b> <span style="color: #e74c3c; font-size: 18px;">{cell_count}</span></p>
        <br>
        <p>Analiz başarıyla tamamlanmıştır.</p>
      </body>
    </html>
    """
    message.attach(MIMEText(body, "html"))

    try:
        # Outlook bağlantısı için TLS şarttır
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.set_debuglevel(1)  # Hata olursa terminalde detay görmek için
        server.starttls()
        server.login(USERNAME, PASSWORD)
        server.send_message(message)
        server.quit()
        return True
    except Exception as e:
        print(f"Outlook SMTP Hatası: {e}")
        return False