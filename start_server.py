#!/usr/bin/env python3
"""
Kanser Hücresi Tespit Sistemi - Server Başlatıcı
"""
import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Gerekli kütüphaneleri kontrol et"""
    try:
        import fastapi
        import uvicorn
        import torch
        import cv2
        import PIL
        print("✓ Tüm gerekli kütüphaneler yüklü")
        return True
    except ImportError as e:
        print(f"❌ Eksik kütüphane: {e}")
        print("Lütfen şu komutu çalıştırın: pip install -r requirements.txt")
        return False

def start_server():
    """API sunucusunu başlat"""
    print("🚀 Kanser Hücresi Tespit Sistemi başlatılıyor...")
    
    # Gerekli kütüphaneleri kontrol et
    if not check_requirements():
        return
    
    # API klasörüne geç
    api_dir = Path(__file__).parent / "api"
    os.chdir(api_dir)
    
    print("📡 API sunucusu başlatılıyor...")
    print("🌐 Frontend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔄 Health Check: http://localhost:8000/health")
    print("\n⏹️  Durdurmak için Ctrl+C tuşlarına basın\n")
    
    # Tarayıcıyı otomatik aç (5 saniye sonra)
    def open_browser():
        time.sleep(5)
        try:
            webbrowser.open("http://localhost:8000")
        except:
            pass
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Sunucuyu başlat
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Sunucu kapatılıyor...")
    except Exception as e:
        print(f"❌ Sunucu başlatma hatası: {e}")

if __name__ == "__main__":
    start_server()