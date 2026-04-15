# Few-Shot Segmentation — Detección de Grietas en Imágenes Radiográficas

Framework de investigación para **segmentación few-shot de grietas** en imágenes radiográficas industriales.

El sistema aprende a segmentar grietas a partir de muy pocos ejemplos anotados (1–5 shots), usando una arquitectura de **Encoder Siamés + Módulo de Prototipos + Decoder U-Net**.

---

## Estructura del proyecto

```
fewshot/
├── config/
│   ├── __init__.py
│   └── base_config.py              ← Dataclasses de configuración global
│
├── datasets/
│   ├── __init__.py
│   ├── episode_dataset.py          ← Dataset episódico (formato .tiff)
│   └── episode_dataset_png.py      ← Dataset episódico (formato .png)
│
├── models/
│   ├── __init__.py
│   ├── fewshot_model.py            ← Modelo completo integrado
│   ├── encoders/
│   │   ├── __init__.py
│   │   └── resnet_encoder.py       ← Backbone ResNet con skip connections
│   ├── fewshot/
│   │   ├── __init__.py
│   │   ├── prototype_module.py     ← Masked Average Pooling → prototipo
│   │   └── similarity.py          ← Mapas de similitud coseno
│   └── decoders/
│       ├── __init__.py
│       └── unet_decoder.py         ← Decoder estilo U-Net
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                  ← Loop de entrenamiento episódico
│   ├── losses.py                   ← Pérdida Dice + BCE ponderada
│   └── metrics.py                  ← IoU, Dice score
│
├── experiments/
│   ├── __init__.py
│   └── baseline.py                 ← Experimento baseline
│
├── utils/
│   └── __init__.py
│
├── imagenes_inferencia/            ← Imágenes de ejemplo para inferencia
│   ├── support_img.png
│   ├── support_mask.png
│   └── query_img.png
│
├── test/
│   └── test_smoke.py               ← Tests de integración básicos
│
├── train.py                        ← Script principal de entrenamiento
├── infer.py                        ← Script de inferencia por parches
└── requirements.txt
```

---

## Arquitectura

```
Imagen soporte ──→ Encoder ──→ Features soporte ──→ Prototipo (grieta + fondo)
                                                            │
Imagen query ───→ Encoder ──→ Features query ──→ Mapas similitud ──→ Decoder ──→ Máscara
                    │                                                      ↑
                    └──────────────── skip connections ───────────────────┘
```

### Componentes clave

| Módulo | Descripción |
|--------|-------------|
| **ResNet Encoder** | Backbone preentrenado (resnet34/50), extrae features multiescala |
| **Prototype Module** | Masked Average Pooling sobre features del soporte → vector prototipo |
| **Similarity** | Similitud coseno entre prototipo y features de la query |
| **U-Net Decoder** | Upsampling con skip connections, produce máscara binaria |

---

## Paradigma de entrenamiento

Entrenamiento episódico — cada episodio contiene:

- `imagen_soporte` + `máscara_soporte` → se usa para calcular el prototipo de grieta
- `imagen_query` + `máscara_query` → el loss se calcula **únicamente** sobre este branch

> **Regla crítica:** La máscara del soporte solo se usa para el cálculo del prototipo.
> El loss **nunca** se aplica sobre la predicción del soporte.

---

## Preprocesamiento de entrada

Cada imagen radiográfica se convierte a un tensor de 3 canales:

| Canal | Descripción | Método |
|-------|-------------|--------|
| 1 | Radiografía normalizada | Clipping percentil 1–99 |
| 2 | Realce de bordes | Unsharp mask |
| 3 | Filtro de alta frecuencia | Diferencia de Gaussianas |

Forma final del tensor: `3 × H × W`

---

## Uso

### Entrenamiento

```bash
python train.py
```

La configuración se gestiona desde `config/base_config.py`: backbone, tamaño de parche, número de epochs, learning rate, etc. Los checkpoints se guardan en `checkpoints/`. **El tamaño de parche con el que se entrena determina el que debe usarse en inferencia.**

### Inferencia sobre imágenes grandes

El script `infer.py` divide la imagen query en parches del mismo tamaño usado durante el entrenamiento, infiere sobre cada uno y reconstruye la máscara a tamaño original. El tamaño de parche y el umbral de decisión se pasan como parámetros según cómo se configuró el entrenamiento.

```bash
python infer.py \
  --support_img imagenes_inferencia/support_img.png \
  --support_mask imagenes_inferencia/support_mask.png \
  --query_img imagenes_inferencia/query_img.png \
  --checkpoint checkpoints/baseline/best_model.pt \
  --output results/ \
  --patch_size <tamaño_de_parche_usado_en_entrenamiento> \
  --threshold <umbral_entre_0_y_1>
```

**Salidas generadas en `results/`:**
- `query.png` — imagen query original
- `pred_mask.png` — máscara de grietas predicha
- `overlay.png` — grietas en rojo sobre la imagen query

---

## Instalación

```bash
pip install -r requirements.txt
```

Dependencias principales:

```
torch>=2.0
torchvision>=0.15
numpy
opencv-python
scikit-image
albumentations
pillow
```

---

## Estado del proyecto

| Componente | Estado |
|---|---|
| Encoder ResNet con skip connections | ✅ |
| Módulo de prototipos (MAP) | ✅ |
| Módulo de similitud coseno | ✅ |
| Decoder U-Net | ✅ |
| Modelo completo integrado | ✅ |
| Dataset episódico (TIFF + PNG) | ✅ |
| Pipeline de entrenamiento | ✅ |
| Inferencia por parches en imágenes grandes | ✅ |
| Tests de integración | ✅ |
