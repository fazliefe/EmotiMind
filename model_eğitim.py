import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import torch

def main():
    # === CUDA Bilgisi ===
    print("CUDA aktif mi:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Kullanılan GPU:", torch.cuda.get_device_name(0))

    # === 1. Veri setini oku ve düzelt ===
    try:
        df = pd.read_csv("veri_setik.csv", encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv("veri_setik.csv", encoding="ISO-8859-9", on_bad_lines="skip")

    # BOM karakteri ve boşluk temizle
    df.columns = [col.strip().lower().replace('\ufeff', '') for col in df.columns]
    print("Sütunlar:", df.columns)

    # ✅ GİRİŞ: DUYGU — ÇIKIŞ: ÖNERİ olarak eşle
    if "oneri" in df.columns and "duygu" in df.columns:
        df = df[["duygu", "oneri"]].dropna()
        df = df.rename(columns={"duygu": "input", "oneri": "target"})

    else:
        raise ValueError("Hata: 'oneri' ve 'duygu' sütunları bulunamadı.")

    print(f"Toplam veri: {len(df)}")
    if len(df) == 0:
        raise ValueError("Hata: Veri seti boş. Lütfen dosyayı kontrol edin.")

    # === 2. Train/Test böl
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
    train_df.to_csv("train_textgen.csv", index=False)
    val_df.to_csv("val_textgen.csv", index=False)

    # === 3. Hugging Face Dataset'e dönüştür
    dataset = DatasetDict({
        "train": load_dataset("csv", data_files={"train": "train_textgen.csv"}, split="train"),
        "validation": load_dataset("csv", data_files={"validation": "val_textgen.csv"}, split="validation")
    })

    # === 4. Model ve Tokenizer
    model_name = "ozcangundes/mt5-small-turkish-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # === 5. Tokenization
    max_input_len = 32
    max_target_len = 64

    def preprocess(example):
        inputs = tokenizer(example["input"], max_length=max_input_len, truncation=True, padding="max_length")
        targets = tokenizer(example["target"], max_length=max_target_len, truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_datasets = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # === 6. Eğitim Ayarları
    training_args = Seq2SeqTrainingArguments(
        output_dir="./textgen_model",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        dataloader_num_workers=0,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    # === 7. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # === 8. Eğitim
    trainer.train()

    # === 9. Kaydet
    model.save_pretrained("./textgen_model")
    tokenizer.save_pretrained("./textgen_model")

# === Windows için şart ===
if __name__ == "__main__":
    main()
