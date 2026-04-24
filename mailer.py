import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_pathovision_report(target_email, label, confidence):
    """
    Analiz sonucunu belirtilen e-posta adresine gönderir.
    """
    # Mailtrap'ten aldığın SMTP bilgileri (Sandbox > SMTP Settings > Python kısmından bakabilirsin)
    SMTP_SERVER = "sandbox.smtp.mailtrap.io"
    SMTP_PORT = 2525
    USERNAME = "5cc3acdc8bb830"  # Mailtrap'teki 14 haneli sayı
    PASSWORD = "68dd4181dd6722"  # Mailtrap'teki 14 haneli şifre

    # E-posta Başlık Bilgileri
    message = MIMEMultipart()
    message["From"] = "reports@pathovision.ai"
    message["To"] = target_email
    message["Subject"] = f"PathoVision AI: Analiz Sonucu - {label}"

    # E-posta Gövdesi (HTML formatında daha profesyonel durur)
    body = f"""
    <html>
      <body style="font-family: sans-serif; border: 1px solid #eee; padding: 20px;">
        <h2 style="color: #2c3e50;">PathoVision AI Analiz Raporu</h2>
        <hr>
        <p><b>Tespit Edilen Durum:</b> <span style="color: #e74c3c;">{label}</span></p>
        <p><b>Güven Oranı:</b> %{confidence}</p>
        <p><b>İşlem Tarihi:</b> 2026-04-24</p>
        <hr>
        <p style="font-size: 12px; color: #7f8c8d;">Bu rapor PathoVision AI sistemi tarafından otomatik olarak oluşturulmuştur.</p>
      </body>
    </html>
    """

    message.attach(MIMEText(body, "html"))

    try:
        # Sunucuya bağlan ve gönder
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.login(USERNAME, PASSWORD)
            server.send_message(message)
        return True
    except Exception as e:
        print(f"E-posta gönderilirken hata oluştu: {e}")
        return False