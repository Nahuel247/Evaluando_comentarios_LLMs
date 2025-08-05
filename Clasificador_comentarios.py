###########################################
# MODELO LLM PARA CLASIFICAR COMENTARIOS
###########################################

# Autor: Nahuel Canelo
# Correo: nahuelcaneloaraya@gmail.com
# Notebook: Notebook Gamer ROG Zephyrus G16 Intel Core Ultra 9 185H NVIDIA GeForce RTX 4070 8GB 16.0" OLED 240Hz 32GB RAM 1TB GU605MI-QR043W

# #########################
# IMPORTAMOS LIBRERÍAS
# #########################

import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from evaluate import load as load_metric
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)


#################################
#  DEFINIMOS FUNCIONES Y CLASES
#################################

# CLASE para guardar métricas
class SaveMetricsCallback(TrainerCallback):
    def __init__(self):
        self.epoch_logs = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log_entry = metrics.copy()
            log_entry["epoch"] = state.epoch
            self.epoch_logs.append(log_entry)

    def save_to_csv(self, path="train_eval_metrics.csv"):
        df = pd.DataFrame(self.epoch_logs)
        df.to_csv(path, index=False)
        print(f"\n Métricas guardadas en {path}")


# Función para tokenizar el texto
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Función para cálcular accuracy por epoca
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.tensor(logits).argmax(dim=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Función para extraer la predicción y labels en train y test
def get_predictions_and_labels(dataset):
    predictions_output = trainer.predict(dataset)
    preds = np.argmax(predictions_output.predictions, axis=1)
    labels = predictions_output.label_ids
    return preds, labels


#################################
#      CARGAMOS LOS DATOS
#################################

# Cargamos base de datos de IBM que contiene comentarios y la etiqueta si el comentario es positivo o negativo
train_data = load_dataset("imdb", split="train[:80%]")
test_data = load_dataset("imdb", split="test[:80%]")  # para acelerar el proceso

#################################
#  TRANSFORMACIÓN DE LA DATA
#################################

# Extracción del método de tokenización utilizada por el modelo
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenización del texto
tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)

# pytorch debe recibir la columna "lebels" para identificarla en el entrenamiento
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")

# Le damos formato correspondiente
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

#######################################
# CARGA DEL MODELO y configurar GPU
#######################################

# Cargamos el modelo y le agregamos una capa nueva que la usaremos para realizar clasificacion
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Solicitamos que trabaje con cuda, si no cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Usando dispositivo: {device}")


##############################################
#  PARAMETRIZACIÓN Y DESEMPEÑO DEL MODELO
##############################################

# Metrica de evaluación
accuracy_metric = load_metric("accuracy")

# Parametrización del modelo
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50
)


#################################
#     ENTRENAMIENTO
#################################

metrics_callback = SaveMetricsCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback]
)

trainer.train()
metrics_callback.save_to_csv("train_eval_metrics.csv")


###################################
# EVALUACIÓN DEL MODELO
###################################
print("\n Evaluación final del modelo entrenado:\n")

# Entrenamiento
train_metrics = trainer.evaluate(tokenized_train)
print(f" Accuracy en entrenamiento: {train_metrics['eval_accuracy']:.4f}")
print(f" Pérdida en entrenamiento: {train_metrics['eval_loss']:.4f}")

# Test
test_metrics = trainer.evaluate(tokenized_test)
print(f"\n Accuracy en test: {test_metrics['eval_accuracy']:.4f}")
print(f" Pérdida en test: {test_metrics['eval_loss']:.4f}")

#######################################################
# 8. Reporte de clasificación y matriz de confusión
#######################################################

# Train
train_preds, train_labels = get_predictions_and_labels(tokenized_train)
print("\n Reporte de clasificación (Train):")
print(classification_report(train_labels, train_preds, labels=[0, 1], target_names=["NEGATIVE", "POSITIVE"]))

# Test
test_preds, test_labels = get_predictions_and_labels(tokenized_test)
print("\n Reporte de clasificación (Test):")
print(classification_report(test_labels, test_preds, labels=[0, 1], target_names=["NEGATIVE", "POSITIVE"]))

# Matriz de confusión
print("\n Matriz de confusión (Test):")
print(confusion_matrix(test_labels, test_preds, labels=[0, 1]))



#################################
# GUARDAMOS EL MODELO
##################################

# Si los resultados son buenos, se guarda el modelo.
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")


#############################
# PRUEBA DE PREDICCIÓN
#############################

sample_texts = [
    "This movie was terrible and boring.",
    "I absolutely loved this film, it was amazing!"
]

inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

labels_map = {0: "NEGATIVE", 1: "POSITIVE"}

for i, pred in enumerate(predictions):
    print(f"\n Review: {sample_texts[i]}")
    print(f" Predicted sentiment: {labels_map[pred.item()]}")
