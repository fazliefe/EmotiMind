# Duygu Tanıma ve Duyguya Dayalı Öneri Sistemi

Bu proje, görseldeki yüz ifadelerini analiz ederek duygu tahmini yapar ve buna uygun Türkçe öneriler üretir. Görüntü işleme (OpenCV), derin öğrenme (TensorFlow), ve doğal dil işleme (Transformers) teknolojilerini bir arada kullanır.

---

## 📁 Proje Klasör Yapısı

### `data/`
Kullanılan veri dosyalarını içerir.

- `mutsuz.jpg` — Örnek test görseli
- `haarcascade_frontalface_default.xml` — OpenCV için yüz tespit modeli

### `model/`
Eğitilmiş modellerin yer aldığı klasör.

- `emotion_model_resnet50_transfer.keras` — Duygu sınıflandırma modeli (Keras formatında)
- `textgen_model/` — HuggingFace'den indirilmiş ve Türkçe öneri üretmek için kullanılan mT5 modeli

### `scripts/`
Tüm Python betikleri bu klasörde yer alır.

- `model_guc.py` — Duygu sınıflandırma modelinin eğitim scripti (ResNet50 transfer learning)
- `main.py` — Ana çalışma dosyası: yüz algılama, duygu tahmini, öneri üretimi ve görsel üzerine yazma işlemleri

### Ana Dizin Dosyaları

- `README.md` — Bu projenin açıklama dosyası
- `requirements.txt` — Projenin bağımlılık listesi (pip ile kurulum için)
- `.gitignore` — Git'e dahil edilmemesi gereken dosya ve klasörleri listeler
- `LICENSE` — MIT lisansı (açık kaynak kullanım şartları)

---

## ⚙️ Kurulum ve Kullanım

```bash
pip install -r requirements.txt
python scripts/main.py

---

🧪 Eğitim Bilgisi
Model, scripts/model_guc.py içinde transfer öğrenme yöntemiyle eğitilmiştir:

Base: ResNet50 (ImageNet ağırlıklı)

Fine-tuning: Son 50 katman açılarak optimize edildi

Çıktı: .keras uzantılı model dosyası

💡 Tanınan Duygular
İngilizce	Türkçe
angry	kızgın
disgust	iğrenme
fear	korku
happy	mutlu
neutral	nötr
sad	üzgün
surprise	şaşkın

📦 Gerekli Kütüphaneler
requirements.txt içeriği:

txt
Kopyala
Düzenle
tensorflow>=2.11
torch>=2.0
transformers>=4.30
opencv-python
numpy

---
🔐 Lisans
MIT Lisansı altında açık kaynak olarak sunulmuştur.
