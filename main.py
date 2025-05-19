import cv2
import tensorflow as tf
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Haar Cascade ile yüz algılama ===
cascade_path = r"haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f"Haar cascade yüklenemedi: {cascade_path}")

# === Duygu sınıflandırma modeli ===
emotion_model = tf.keras.models.load_model("emotion_model_resnet50_transfer.keras")
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === mt5 tabanlı öneri üretim modeli ===
textgen_model_path = "./textgen_model"
tokenizer = AutoTokenizer.from_pretrained(textgen_model_path)
textgen_model = AutoModelForSeq2SeqLM.from_pretrained(textgen_model_path)

# === İngilizce → Türkçe duygu dönüşümü ===
emotion_to_text = {
    "angry": "kızgın",
    "disgust": "iğrenme",
    "fear": "korku",
    "happy": "mutlu",
    "neutral": "nötr",
    "sad": "üzgün",
    "surprise": "şaşkın"
}

# === Yedek öneriler (model başarısız olursa) ===
yedek_oneriler = {
    "kızgın": "Sakin ol, derin bir nefes al.",
    "üzgün": "Bu his geçecek. Yalnız değilsin.",
    "mutlu": "Harika! Bu anın tadını çıkar.",
    "nötr": "Her şey yolunda, böyle devam et.",
    "şaşkın": "Beklenmedik şeyler de güzeldir bazen.",
    "korku": "Korkular geçicidir. Güvendesin.",
    "iğrenme": "Bazen bazı şeyleri kabul etmek zordur, ama geçer."
}

# === Öneri üretme fonksiyonu ===
def oneri_ver(emotion):
    duygu = emotion_to_text.get(emotion.lower(), "")
    if not duygu:
        return "Tanımsız duygu."

    girdi = f"{duygu} hissediyorum"
    inputs = tokenizer(girdi, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        output = textgen_model.generate(
            inputs["input_ids"],
            max_length=50,
            num_beams=4,
            early_stopping=True
        )

    sonuc = tokenizer.decode(output[0], skip_special_tokens=True)
    if not sonuc.strip():
        return yedek_oneriler.get(duygu, "Kendini dinle, her şey geçici.")
    return sonuc

# === Görsel oku ===
img_path = "mutsuz.jpg"  # Değiştirilebilir
img = cv2.imread(img_path)
if img is None:
    raise IOError(f"Görsel bulunamadı: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

def preprocess_face(face_img):
    img = cv2.resize(face_img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === Her yüz için duygu tahmini ve öneri üretimi ===
for (x, y, w, h) in faces:
    face_img = img[y:y+h, x:x+w]
    input_img = preprocess_face(face_img)

    preds = emotion_model.predict(input_img)[0]
    idx = np.argmax(preds)
    emotion = emotions[idx]
    confidence = preds[idx]

    print("------")
    print("Duygu:", emotion)
    print("Güven: %.2f%%" % (confidence * 100))

    # 🧠 NLP öneri üret
    oneri = oneri_ver(emotion)
    print("🧠 NLP Öneri:", oneri)
    print("------")

    # Görsel üzerine çizim
    label = f"{emotion} ({confidence*100:.1f}%)"
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(img, oneri, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

# === Sonuçları göster ===
cv2.imshow("Duygu ve NLP Öneri", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
