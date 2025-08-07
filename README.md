# Clasificación de Comentarios con un Modelo LLM (BERT + Hugging Face)

Este proyecto se entrena e implementa un modelo de lenguaje (LLM) para clasificar comentarios como positivos o negativos, utilizando el dataset de IMDb para entrenar al modelo. Se hace uso de la librería `transformers` de Hugging Face y PyTorch para realizar el fine-tuning del modelo `bert-base-uncased`.

## Objetivo

Demostrar cómo entrenar un modelo de clasificación de texto utilizando técnicas modernas de NLP con un enfoque práctico y accesible. El proyecto incluye desde la carga de datos hasta la evaluación e inferencia del modelo entrenado.

## Requisitos Previos

Para entender y ejecutar este proyecto, es importante manejar los siguientes conceptos:

- **Transfer Learning**: Utilización de modelos preentrenados como BERT y adaptación a tareas específicas.
- **Tokenización**: Conversión del texto a secuencias numéricas mediante `AutoTokenizer`.
- **Fine-Tuning**: Ajuste de los pesos del modelo con un dataset etiquetado.
- **Métricas y Callbacks**: Registro del rendimiento durante el entrenamiento mediante `TrainerCallback`.
- **Inferencia**: Clasificación de nuevos comentarios una vez entrenado el modelo.

## Detalles Técnicos

- **Modelo**: `bert-base-uncased`
- **Dataset**: IMDb (disponible en Hugging Face Datasets)
- **Frameworks**: PyTorch, Hugging Face Transformers, Datasets y Evaluate
- **Entrenamiento**: Incluye soporte para GPU
- **Evaluación**: Accuracy, pérdida, reporte de clasificación, matriz de confusión
- **Exportación**: Guarda modelo y tokenizer fine-tuneados para uso posterior


Los resultados en entrenamiento y testeo muestran una alta capacidad del modelo para identificar si un comentario es positivo o negativo.

![4 report](https://github.com/user-attachments/assets/4435ea33-3013-485e-bc6c-a014f5a71c5c)

## Archivos Incluidos

- `main.py`: Script principal con todo el flujo (entrenamiento, evaluación e inferencia)
- `train_eval_metrics.csv`: Métricas por época
- `fine_tuned_model/`: Carpeta generada con el modelo y tokenizer entrenados

## Ejemplo de Inferencia

Una vez entrenado el modelo, puedes clasificar nuevos comentarios como se muestra en el ejemplo al final del script. Se utiliza `argmax` sobre los logits para obtener la clase más probable (positiva o negativa).

## Requisitos

- Python 3.8+
- torch
- transformers
- datasets
- evaluate
- scikit-learn
- pandas

Puedes instalar las dependencias necesarias con:

```bash
pip install torch transformers datasets evaluate scikit-learn pandas
