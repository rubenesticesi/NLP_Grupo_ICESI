# 🎬 Clasificación de Críticas de Cine con Embeddings y Redes Recurrentes

Proyecto académico de **Procesamiento de Lenguaje Natural (NLP)** orientado a la clasificación binaria de sentimiento en críticas cinematográficas del dataset **IMDb Large Movie Review Dataset**.

El cuaderno compara tres arquitecturas recurrentes implementadas con **PyTorch** y añade una etapa de exploración semántica utilizando embeddings preentrenados de **spaCy**.

El objetivo principal es estudiar cómo la **bidireccionalidad** y un **mecanismo de atención de Bahdanau** afectan el rendimiento, la eficiencia y la interpretabilidad de un clasificador de sentimiento, manteniendo un perfil de ejecución viable en CPU.

---

## 📌 Contexto académico

* **Universidad:** ICESI
* **Curso:** NLP — Procesamiento de Lenguaje Natural
* **Profesor:** Luis Ferro Diez
* **Periodo:** Tercer semestre de 2026
* **Autores:** Edwin Perez, Rubén Darío Sabogal y Cristian Camilo Quebrada

---

## 🎯 Objetivos

Este notebook permite:

* Explorar embeddings preentrenados de **spaCy `en_core_web_lg`** aplicados al vocabulario cinematográfico.
* Analizar similitud coseno, analogías vectoriales y agrupaciones semánticas mediante **t-SNE**.
* Realizar análisis exploratorio del dataset IMDb.
* Construir un pipeline reproducible de tokenización, vocabulario y carga de datos.
* Entrenar y comparar tres modelos recurrentes:

  1. **LSTM unidireccional**.
  2. **GRU bidireccional**.
  3. **BiLSTM bidireccional con atención de Bahdanau**.
* Evaluar los modelos mediante Accuracy, Precision, Recall, F1 y ROC-AUC.
* Analizar el equilibrio entre rendimiento y número de parámetros.
* Visualizar los pesos de atención para interpretar las decisiones del mejor modelo.
* Ejecutar predicciones sobre críticas nuevas.

---

# 🧠 Modelos evaluados

## 1. LSTM Simple

Arquitectura base:

```text
Tokens
   ↓
Embedding
   ↓
LSTM
   ↓
h_T
   ↓
Dropout
   ↓
Linear
   ↓
Sentimiento
```

La red procesa la secuencia de izquierda a derecha y utiliza el último estado oculto como representación global del texto.

---

## 2. GRU Bidireccional

```text
Tokens
   ↓
Embedding
   ↓
BiGRU
   ↓
[h_forward ; h_backward]
   ↓
Dropout
   ↓
Linear
   ↓
Sentimiento
```

La bidireccionalidad permite incorporar contexto anterior y posterior antes de realizar la clasificación.

---

## 3. BiLSTM + Atención de Bahdanau

```text
Tokens
   ↓
Embedding
   ↓
BiLSTM
   ↓
Estados ocultos de toda la secuencia
   ↓
Atención de Bahdanau
   ↓
Vector de contexto ponderado
   ↓
LayerNorm
   ↓
MLP
   ↓
Sentimiento
```

El mecanismo de atención calcula un peso para cada posición de la secuencia:

```text
score(t) = vᵀ · tanh(W · h_t + b)

α_t = softmax(score(t))

contexto = Σ α_t · h_t
```

Esto permite que el modelo utilice una representación ponderada de toda la reseña y, además, posibilita inspeccionar qué palabras tuvieron mayor influencia en la predicción.

> **Nota técnica:** los vectores preentrenados de spaCy se utilizan en la sección exploratoria del notebook. Los tres clasificadores recurrentes emplean su propia capa `nn.Embedding` entrenable de **64 dimensiones**.

---

# 📊 Dataset

Se utiliza el **IMDb Large Movie Review Dataset**, cargado mediante Hugging Face Datasets:

```python
from datasets import load_dataset

imdb = load_dataset("stanfordnlp/imdb")
```

El dataset IMDb es un benchmark ampliamente utilizado para análisis de sentimiento y contiene críticas cinematográficas etiquetadas como:

```text
0 → Negativo
1 → Positivo
```

El notebook selecciona subconjuntos estratificados para controlar el costo computacional.

---

## ⚙️ Perfil CPU utilizado

En la ejecución almacenada en el notebook se utilizaron:

| Conjunto                             | Registros |
| ------------------------------------ | --------: |
| Subconjunto inicial de entrenamiento |     3.000 |
| Entrenamiento efectivo               |     2.700 |
| Validación                           |       300 |
| Prueba                               |     1.000 |
| Positivos                            |     1.500 |
| Negativos                            |     1.500 |

Cuando se detecta una GPU, el notebook aumenta automáticamente los subconjuntos a:

```text
12.000 críticas de entrenamiento
3.000 críticas de prueba
```

---

# 🔎 Exploración de embeddings

Antes de construir los clasificadores se utiliza:

```text
spaCy en_core_web_lg
```

El modelo dispone de vectores densos de:

```text
300 dimensiones
```

En la ejecución registrada se detectaron:

```text
342.918 palabras × 300 dimensiones
```

La exploración incluye:

* matriz de similitud coseno;
* vocabulario positivo y negativo;
* roles cinematográficos;
* analogías vectoriales;
* reducción dimensional mediante t-SNE;
* visualización de clusters semánticos.

Ejemplos de términos analizados:

```text
brilliant
masterpiece
outstanding
excellent
wonderful

terrible
awful
boring
dreadful
disappointing

actor
actress
director
producer

comedy
thriller
horror
drama
romance
```

---

# 🔬 Análisis exploratorio de datos — EDA

El notebook analiza:

* distribución de longitud de las críticas;
* diferencias entre críticas positivas y negativas;
* percentiles de longitud;
* distribución acumulada;
* palabras más frecuentes;
* vocabulario característico por sentimiento;
* nubes de palabras.

En la muestra utilizada:

| Sentimiento | Media de palabras | Mediana | Máximo |
| ----------- | ----------------: | ------: | -----: |
| Negativo    |             226,7 |     172 |  1.014 |
| Positivo    |             239,0 |     171 |  1.527 |

Este análisis sirve como fundamento para seleccionar la longitud máxima de las secuencias.

---

# ⚙️ Pipeline de procesamiento

El procesamiento de texto implementado incluye:

1. Eliminación de etiquetas HTML.
2. Conversión a minúsculas.
3. Conservación de caracteres alfabéticos.
4. Tokenización.
5. Construcción del vocabulario.
6. Uso de `[PAD]`.
7. Uso de `[UNK]`.
8. Truncamiento de secuencias.
9. Padding dinámico.
10. División estratificada entrenamiento/validación.

Ejemplo de tokenización:

```python
def tokenizar(texto):
    texto = re.sub(r'<[^>]+>', ' ', texto)
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto.split()
```

---

# ⚙️ Hiperparámetros principales

| Parámetro          |            Valor |
| ------------------ | ---------------: |
| Semilla global     |               42 |
| Vocabulario máximo |           15.000 |
| Longitud máxima    |       160 tokens |
| Embedding          |               64 |
| Estado oculto      |               96 |
| Batch CPU          |               16 |
| Batch GPU          |               64 |
| Dropout            |             0,25 |
| Épocas máximas     |                5 |
| Learning rate      |            0,001 |
| Optimizador        |             Adam |
| Weight decay       |             1e-5 |
| Loss               | CrossEntropyLoss |
| Gradient clipping  |              1,0 |

El vocabulario final obtenido en la ejecución fue:

```text
15.000 tokens
```

a partir de aproximadamente:

```text
30.161 tokens únicos
```

---

# 🛠️ Estrategia de entrenamiento

Los tres modelos utilizan una infraestructura común para garantizar una comparación metodológicamente consistente.

Incluye:

### Early Stopping

Detiene el entrenamiento cuando `val_loss` deja de mejorar.

### Gradient Clipping

```python
nn.utils.clip_grad_norm_(
    modelo.parameters(),
    max_norm=1.0
)
```

Ayuda a controlar el problema de gradientes explosivos frecuente en redes recurrentes.

### Scheduler

Se emplea:

```python
torch.optim.lr_scheduler.ReduceLROnPlateau
```

para disminuir automáticamente el learning rate cuando la pérdida de validación deja de mejorar.

### Restauración del mejor modelo

Al finalizar el entrenamiento se restauran los parámetros correspondientes al mejor `val_loss`.

---

# 📈 Resultados experimentales

Los resultados almacenados actualmente en el notebook son:

| Modelo                |    Parámetros |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |  Test Loss |
| --------------------- | ------------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| LSTM Simple           |     1.022.402 |     0,5870 |     0,5912 |     0,5640 |     0,5773 |     0,6347 |     0,6768 |
| GRU Bidireccional     |     1.053.698 |     0,6880 |     0,6822 |     0,7040 |     0,6929 |     0,7370 |     0,6143 |
| **BiLSTM + Atención** | **1.140.770** | **0,7650** | **0,7379** | **0,8220** | **0,7777** | **0,8492** | **0,4877** |

---

# 🏆 Mejor modelo

El mejor modelo de la ejecución experimental fue:

## BiLSTM + Atención de Bahdanau

Resultados:

```text
Accuracy ............ 76,50 %
Precision ........... 73,79 %
Recall .............. 82,20 %
F1 .................. 0,7777
ROC-AUC ............. 0,8492
Test Loss ........... 0,4877
Mejor Val Accuracy .. 77,00 %
```

La comparación respalda la hipótesis experimental planteada en el notebook:

> **Una BiLSTM con mecanismo de atención puede superar a una LSTM unidireccional simple en la clasificación de sentimiento de críticas cinematográficas.**

---

# 📊 Evolución del rendimiento

### LSTM Simple

```text
Test Accuracy: 58,70 %
ROC-AUC:       0,6347
```

### GRU Bidireccional

```text
Test Accuracy: 68,80 %
ROC-AUC:       0,7370
```

### BiLSTM + Atención

```text
Test Accuracy: 76,50 %
ROC-AUC:       0,8492
```

La evolución observada fue:

```text
LSTM
58,7 %
   │
   ▼
GRU Bidireccional
68,8 %
   │
   ▼
BiLSTM + Atención
76,5 %
```

---

# 🔍 Interpretabilidad mediante atención

Una de las principales ventajas de la BiLSTM es la posibilidad de analizar los pesos generados por el mecanismo de atención.

El notebook implementa:

```python
obtener_atención(texto)
```

y:

```python
visualizar_atención(texto)
```

Esto permite identificar las palabras con mayor influencia sobre cada clasificación.

En diferentes ejemplos experimentales aparecieron entre los tokens de mayor atención:

```text
best
movies
brilliant
absolute
terrible
completely
acting
```

---

# 🧠 ¿Por qué es importante la atención?

Los modelos LSTM y GRU convencionales tienden a condensar la información de la secuencia en una representación limitada.

La atención permite calcular una representación:

```text
contexto = Σ α_t h_t
```

donde:

```text
α_t
```

representa la importancia relativa del token situado en la posición `t`.

Esto facilita:

* interpretación;
* depuración;
* análisis de errores;
* explicación de predicciones;
* detección de posibles sesgos.

---

# ⚠️ Caso difícil: ironía

El notebook también analiza expresiones irónicas.

Ejemplo conceptual:

```text
Oh yes, another masterpiece from Hollywood...
```

El modelo puede asignar mucha importancia a palabras aparentemente positivas como:

```text
masterpiece
```

sin interpretar correctamente la intención sarcástica.

Esto muestra una limitación relevante de los modelos recurrentes utilizados.

---

# 🧪 Predicciones sobre nuevas críticas

El notebook incluye:

```python
predecir_sentimiento(texto)
```

Ejemplo:

```python
resultado = predecir_sentimiento(
    "This film is brilliant and the performances are outstanding."
)

print(resultado)
```

La respuesta tiene una estructura equivalente a:

```python
{
    "sentimiento": "Positivo",
    "prob_negativo": ...,
    "prob_positivo": ...,
    "confianza": ...
}
```

---

# 🎥 Ejemplos incluidos

Entre las frases utilizadas para evaluar el modelo aparecen:

```text
The Shawshank Redemption is a timeless classic.
Morgan Freeman is perfect.
```

Resultado registrado:

```text
Positivo
Confianza: 0,845
```

También:

```text
I fell asleep halfway through this boring disaster.
The script is terrible.
```

Resultado:

```text
Negativo
Confianza: 0,955
```

---

# 📊 Métricas utilizadas

La evaluación considera:

### Accuracy

Proporción global de predicciones correctas.

### Precision

```text
TP / (TP + FP)
```

Mide qué proporción de predicciones positivas es realmente positiva.

### Recall

```text
TP / (TP + FN)
```

Mide qué porcentaje de casos positivos logra identificar el modelo.

### F1-score

Media armónica entre Precision y Recall:

```text
F1 = 2 × Precision × Recall
         ──────────────────
         Precision + Recall
```

### ROC-AUC

Evalúa la capacidad del clasificador para separar ambas clases a diferentes umbrales.

---

# 📉 Matrices de confusión

El notebook genera matrices de confusión normalizadas para:

```text
LSTM Simple
GRU Bidireccional
BiLSTM + Atención
```

Las filas corresponden a las clases reales y las columnas a las predicciones.

```text
                  Predicción
              Negativo  Positivo

Real Negativo     ✓         ✗
Real Positivo     ✗         ✓
```

---

# ⚖️ Eficiencia de parámetros

El notebook analiza también el trade-off:

```text
Complejidad del modelo
        vs.
Rendimiento
```

Número de parámetros:

```text
LSTM Simple
1.022.402

GRU Bidireccional
1.053.698

BiLSTM + Atención
1.140.770
```

Aunque BiLSTM + Atención es el modelo de mayor tamaño, la mejora en rendimiento observada justifica experimentalmente su mayor complejidad dentro de este ejercicio.

---

# 💻 Perfil optimizado para CPU

El notebook fue diseñado para poder ejecutarse en un computador personal sin TPU.

Configuración:

```text
3.000 reseñas iniciales de entrenamiento
1.000 reseñas de prueba

Vocabulario máximo:
15.000

Longitud:
160 tokens

Embeddings:
64 dimensiones

Estado oculto:
96

Batch:
16

Máximo:
5 épocas
```

También se utiliza padding dinámico para reducir operaciones innecesarias.

---

# 🖥️ Detección automática del hardware

El cuaderno verifica automáticamente:

```python
torch.cuda.is_available()
```

Si existe GPU:

```text
device = cuda
```

En caso contrario:

```text
device = cpu
```

En CPU también limita el número de hilos:

```python
torch.set_num_threads(
    min(4, os.cpu_count() or 1)
)
```

---

# 🔁 Reproducibilidad

Se utiliza una semilla global:

```python
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

Esto ayuda a obtener experimentos más consistentes entre ejecuciones.

---

# 💻 Requisitos

El proyecto utiliza principalmente:

```text
Python 3
PyTorch
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
SciPy
spaCy
datasets
wordcloud
Jupyter Notebook
```

También requiere:

```text
en_core_web_lg
```

---

# 🚀 Instalación local

Se recomienda utilizar un entorno virtual.

```bash
python -m venv .venv
```

## Windows

```bash
.venv\Scripts\activate
```

## Linux/macOS

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy spacy datasets wordcloud jupyter
```

Descargar el modelo de spaCy:

```bash
python -m spacy download en_core_web_lg
```

Iniciar Jupyter:

```bash
jupyter notebook
```

---

# 📓 Notebook principal

El archivo utilizado actualmente es:

```text
embeddings_lstm_criticas_cine_cpu (1).ipynb
```

Para publicarlo en GitHub se recomienda renombrarlo como:

```text
embeddings_lstm_criticas_cine_cpu.ipynb
```

---

# ☁️ Google Colab

El notebook detecta si está siendo ejecutado en Google Colab.

Cuando Colab está disponible instala automáticamente las dependencias principales y descarga:

```text
en_core_web_lg
```

La ejecución debe realizarse secuencialmente, ya que las secciones posteriores dependen del vocabulario, los DataLoaders y los modelos definidos previamente.

---

# 📁 Estructura básica recomendada del repositorio

```text
clasificacion-sentimiento-lstm/
│
├── README.md
│
└── embeddings_lstm_criticas_cine_cpu.ipynb
```

Opcionalmente puede evolucionar a:

```text
clasificacion-sentimiento-lstm/
│
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   └── embeddings_lstm_criticas_cine_cpu.ipynb
├── images/
│   ├── comparacion_modelos.png
│   ├── matriz_confusion.png
│   └── attention_example.png
└── src/
```

---

# 🧭 Flujo general del notebook

```text
Preparación del entorno
        │
        ▼
Exploración de embeddings spaCy
        │
        ▼
EDA del dataset IMDb
        │
        ▼
Tokenización
        │
        ▼
Construcción del vocabulario
        │
        ▼
DataLoaders + padding dinámico
        │
        ├──────────────┐
        ▼              ▼
   LSTM Simple    GRU Bidireccional
        │              │
        └──────┬───────┘
               ▼
      BiLSTM + Atención
               │
               ▼
       Evaluación comparativa
               │
               ▼
      Matrices de confusión
               │
               ▼
       Pesos de atención
               │
               ▼
        Demo de inferencia
               │
               ▼
          Conclusiones
```

---

# 📌 Principales conclusiones

1. La **LSTM simple** proporciona una línea base clara, pero obtuvo el menor rendimiento de los tres modelos.

2. La **GRU bidireccional** mejora significativamente los resultados al incorporar contexto en ambas direcciones.

3. La **BiLSTM con atención de Bahdanau** obtuvo el mejor desempeño experimental.

4. El mecanismo de atención permite identificar qué palabras influyeron más en cada clasificación.

5. La bidireccionalidad mejora la representación del contexto lingüístico.

6. La atención ayuda a evitar que toda la información de una crítica deba condensarse exclusivamente en un único estado oculto.

7. Es posible desarrollar este experimento en un computador sin TPU mediante un perfil reducido y optimizado para CPU.

8. Expresiones como ironía, sarcasmo y ambigüedad siguen representando un desafío.

---

# 🔮 Posibles mejoras

Como continuación del proyecto podrían incorporarse:

* embeddings preentrenados directamente en los clasificadores;
* Word2Vec;
* GloVe;
* FastText;
* mayor número de críticas de entrenamiento;
* búsqueda sistemática de hiperparámetros;
* validación cruzada;
* regularización adicional;
* persistencia del modelo entrenado;
* interfaz para inferencia;
* modelos Transformer;
* BERT;
* DistilBERT;
* análisis formal de errores;
* evaluación de sarcasmo e ironía;
* comparación CPU vs GPU.

---

# 📚 Referencias conceptuales

El notebook se fundamenta principalmente en:

* **IMDb Large Movie Review Dataset** — Maas et al., 2011.
* **GRU** — Chung et al., 2014.
* **Atención aditiva** — Bahdanau et al., 2015.
* **spaCy `en_core_web_lg`** para exploración semántica mediante embeddings.
* **PyTorch** para implementación de redes neuronales recurrentes.

---

# 🎓 Universidad ICESI

**Procesamiento de Lenguaje Natural — NLP**

Proyecto académico para el estudio experimental de:

```text
Word Embeddings
+
LSTM
+
GRU
+
BiLSTM
+
Mecanismos de Atención
+
Análisis de Sentimiento
+
Interpretabilidad
```
