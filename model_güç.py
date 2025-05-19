import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === ✅ TensorFlow GPU yapılandırması ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU kullanılacak:", gpus)
    except RuntimeError as e:
        print("GPU ayarlanamadı:", e)
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Görüntü boyutu ve batch size
img_size = 224
batch_size = 32

# 🔧 Doğru eğitim klasörünü kullan!
data_dir = "veri_seti/train"

# Veri artırımı ve doğrulama ayrımı
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Eğitim verisi
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Doğrulama verisi
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# ✅ Sınıf sıralamasını yazdır (etiket kontrolü için)
print("Sınıf etiket sıralaması:", train_generator.class_indices)

# Önceden eğitilmiş ResNet50 (üst katman yok)
base_model = ResNet50(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
base_model.trainable = False  # İlk aşamada dondur

# Yeni sınıflandırıcı blok
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Model oluştur
model = Model(inputs=base_model.input, outputs=predictions)

# Derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Özet
model.summary()

# İlk eğitim (yalnızca üst katmanlar)
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)

# Fine-tuning: son 50 katmanı aç
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Düşük öğrenme hızı ile yeniden derle
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tuning eğitimi
history_fine = model.fit(
    train_generator,
    epochs=25,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator
)

# ✅ Eğitilen modeli kaydet
model.save("emotion_model_resnet50_transfer.keras")
