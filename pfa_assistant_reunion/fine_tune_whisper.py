import torch
import dataclasses
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

# --- 1. Configuration ---
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "French"
TASK = "transcribe"
# LIGNE CORRIGÉE : Nous utilisons le dataset "fleurs" de Google, beaucoup plus stable.
DATASET = "google/fleurs" 
OUTPUT_DIR = "./fine_tuned_model"

# --- 2. Chargement du processeur et du modèle ---
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# --- 3. Chargement et préparation du dataset ---
common_voice = DatasetDict()

# LIGNE CORRIGÉE : On charge le dataset "fleurs" avec la configuration "fr_fr".
common_voice["train"] = load_dataset(DATASET, "fr_fr", split="train").select(range(100))
common_voice["test"] = load_dataset(DATASET, "fr_fr", split="validation").select(range(50)) # Fleurs utilise 'validation' pour le test

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # LIGNE CORRIGÉE : La colonne de texte s'appelle "raw_transcription".
    batch["labels"] = processor.tokenizer(batch["raw_transcription"]).input_ids
    return batch

print("Préparation du dataset...")
# On supprime les colonnes après le mapping pour plus de simplicité
common_voice = common_voice.map(prepare_dataset, num_proc=1)
print("Préparation terminée.")


# --- 4. Configuration de l'entraînement ---
@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any
    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=200,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# --- 5. Lancement de l'entraînement ---
print("\nLancement du fine-tuning...")
trainer.train()
print("Fine-tuning terminé.")

# --- 6. Sauvegarde du modèle ---
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Modèle sauvegardé dans {OUTPUT_DIR}")