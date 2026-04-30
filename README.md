# Few-Shot Segmentation — Detección de Defectos en Imágenes Industriales

Framework para **segmentación few-shot de defectos** en imágenes industriales.

El sistema aprende a segmentar defectos a partir de muy pocos ejemplos anotados (1–5 shots), usando una arquitectura de **Encoder Siamés + Módulo de Prototipos + Decoder U-Net**. Está diseñado para ser agnóstico al dominio — aunque el caso de uso inicial son grietas en imágenes radiográficas, puede adaptarse a cualquier tipo de defecto con mínimos cambios.

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
│   ├── episode_dataset.py          ← Dataset episódico (formato .tiff, 16bit)
│   └── episode_dataset_png.py      ← Dataset episódico (formato .png, 8bit)
│
├── models/
│   ├── __init__.py
│   ├── fewshot_model.py            ← Modelo completo integrado
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── base_encoder.py         ← Contrato abstracto para todos los encoders
│   │   ├── resnet_encoder.py       ← Backbone ResNet con skip connections
│   │   ├── swin_encoder.py         ← Backbone Swin Transformer via timm
│   │   └── encoder_factory.py      ← Factory — build_encoder(cfg)
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
├── test/
│   └── test_smoke.py               ← Tests de integración básicos
│
├── train.py                        ← Script principal de entrenamiento
├── infer.py                        ← Script de inferencia por parches
└── visualize_predictions.py        ← Visualización de predicciones sobre val set
```

---

## Arquitectura

```
Imagen soporte ──→ Encoder ──→ Features soporte ──→ Prototipo (defecto + fondo)
                                                            │
Imagen query ───→ Encoder ──→ Features query ──→ Mapas similitud ──→ Decoder ──→ Máscara
                    │                                                      ↑
                    └──────────────── skip connections ───────────────────┘
```

### Componentes clave

| Módulo | Descripción |
|--------|-------------|
| **BaseEncoder** | Contrato abstracto — cualquier encoder debe implementarlo |
| **ResNetEncoder** | Backbone torchvision (resnet18/34/50/101), extrae features multiescala |
| **SwinEncoder** | Backbone Swin Transformer via timm, soporta cualquier modelo con features_only |
| **EncoderFactory** | `build_encoder(cfg)` — único punto de entrada para instanciar encoders |
| **Prototype Module** | Masked Average Pooling sobre features del soporte → vector prototipo |
| **Similarity** | Similitud coseno entre prototipo y features de la query |
| **U-Net Decoder** | Upsampling con skip connections, produce máscara binaria |

---

## Backbones soportados

### ResNet (torchvision)

| Backbone | Canales layer4 | Skip channels (l3/l2/l1) |
|----------|---------------|--------------------------|
| resnet18 | 512 | 256 / 128 / 64 |
| resnet34 | 512 | 256 / 128 / 64 |
| resnet50 | 2048 | 1024 / 512 / 256 |
| resnet101 | 2048 | 1024 / 512 / 256 |

### Swin Transformer (timm)

| Backbone | Canales layer4 | Skip channels (l3/l2/l1) |
|----------|---------------|--------------------------|
| swin_tiny_patch4_window7_224 | 768 | 384 / 192 / 96 |
| swin_small_patch4_window7_224 | 768 | 384 / 192 / 96 |
| swin_base_patch4_window7_224 | 1024 | 512 / 256 / 128 |
| swin_large_patch4_window7_224 | 1536 | 768 / 384 / 192 |

Cualquier backbone de timm que soporte `features_only=True` funciona — ConvNeXt, EfficientNet, etc.

---

## Paradigma de entrenamiento

Entrenamiento episódico — cada episodio contiene:

- `imagen_soporte` + `máscara_soporte` → se usa para calcular el prototipo de defecto
- `imagen_query` + `máscara_query` → el loss se calcula **únicamente** sobre este branch

> **Regla crítica:** La máscara del soporte solo se usa para el cálculo del prototipo.
> El loss **nunca** se aplica sobre la predicción del soporte.

---

## Formatos de imagen soportados

| Formato | Profundidad | Dataset class |
|---------|-------------|---------------|
| PNG | 8 bit | `EpisodicDatasetPNG` |
| TIFF | 16 bit | `EpisodicDataset` |

Ambos se normalizan a float32 en [0, 1] antes de entrar al modelo. El backbone no distingue el formato de origen.

---

## Uso

### Entrenamiento

```bash
# PNG 8bit + ResNet34 (default)
python train.py --format png --backbone resnet34 --data data/ --epochs 100

# TIFF 16bit + Swin-Tiny
python train.py --format tiff --backbone swin_tiny_patch4_window7_224 --data data/ --epochs 100

# Con más opciones
python train.py \
  --format png \
  --backbone swin_tiny_patch4_window7_224 \
  --data data/ \
  --epochs 100 \
  --lr 1e-4 \
  --device cuda \
  --batch_size 4
```

Los checkpoints se guardan en `checkpoints/` con su JSON de configuración completo.

### Inferencia sobre imágenes grandes

```bash
python infer.py \
  --support_img imagenes_inferencia/support_img.png \
  --support_mask imagenes_inferencia/support_mask.png \
  --query_img imagenes_inferencia/query_img.png \
  --checkpoint checkpoints/baseline/best_model.pt \
  --output results/ \
  --threshold 0.5
```

**Salidas en `results/`:**
- `query.png` — imagen query original
- `pred_mask.png` — máscara de defectos predicha
- `overlay.png` — defectos en rojo sobre la imagen query

### Visualización de predicciones

```bash
python visualize_predictions.py \
  --checkpoint checkpoints/baseline/best_model.pt \
  --data data/ \
  --n_episodes 8 \
  --threshold 0.5 \
  --format png \
  --output results/viz.png
```

Genera una grilla PNG con columnas: support | query | ground truth | predicción.

---

## Instalación

```bash
pip install -r requirements.txt
```

Dependencias principales:

```
torch>=2.0
torchvision>=0.15
timm>=0.9
numpy
pillow
albumentations
matplotlib
```

---

## Tests

```bash
python -m pytest test/test_smoke.py -v
```

25 tests — cubre todos los módulos incluyendo ResNet y Swin end-to-end.

---

