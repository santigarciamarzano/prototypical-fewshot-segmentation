# Guía de Entrenamiento — Prototypical Few-Shot Segmentation

Paso a paso completo para entrenar el modelo desde cero en un nuevo dominio o dataset.

---

## Índice

1. [Preparar el dataset](#1-preparar-el-dataset)
2. [Elegir el backbone](#2-elegir-el-backbone)
3. [Elegir el formato de imagen](#3-elegir-el-formato-de-imagen)
4. [Configurar el entrenamiento](#4-configurar-el-entrenamiento)
5. [Lanzar el entrenamiento](#5-lanzar-el-entrenamiento)
6. [Monitorear el entrenamiento](#6-monitorear-el-entrenamiento)
7. [Interpretar el checkpoint](#7-interpretar-el-checkpoint)
8. [Inspección visual de predicciones](#8-inspección-visual-de-predicciones)
9. [Inferencia sobre imágenes completas](#9-inferencia-sobre-imágenes-completas)

---

## 1. Preparar el dataset

### Estructura de directorios

```
data/
├── train/
│   ├── images/   imagen_001.png, imagen_002.png, ...   o tiff
│   └── masks/    imagen_001.png, imagen_002.png, ...
└── val/
    ├── images/
    └── masks/
```

### Reglas obligatorias

- **Mismo nombre** para imagen y máscara — `crack_001.png` en images debe tener `crack_001.png` en masks.
- **Mismo formato** en todo el split — no mezclar PNG y TIFF en el mismo directorio.
- **Máscaras binarias** — solo dos valores posibles:
  - PNG 8bit: fondo = 0, defecto = 255
  - TIFF 16bit: fondo = 0, defecto = 65535
- **Mínimo de imágenes** — el dataset necesita al menos `k_shot + 1` imágenes por split. Con k_shot=1 el mínimo es 2 imágenes.

### Tamaño de los parches

Las imágenes deben ser parches del tamaño que se usará en entrenamiento (`image_size` en config). El tamaño más común es 256×256 o 512×512. **Este tamaño debe coincidir con el que se usa en inferencia.**

Si las imágenes originales son más grandes, hay que parchearlas antes de construir el dataset. El script `infer.py` ventanea automáticamente durante inferencia usando el mismo tamaño.

---

## 2. Elegir el backbone

El backbone es la red que extrae features de las imágenes. Hay dos familias disponibles:

### ResNet (torchvision) 

Buena elección si:
- El dataset es pequeño (< 500 imágenes)
- Se quiere entrenamiento rápido
- Es la primera prueba en un dominio nuevo

```bash
--backbone resnet34    # recomendado para empezar
--backbone resnet50    # más capacidad, más lento
```

### Swin Transformer (timm) 

Buena elección si:
- El dataset es mediano o grande
- Los defectos tienen estructura global (grietas largas, patrones repetitivos)
- Se dispone de GPU con al menos 16GB VRAM

```bash
--backbone swin_tiny_patch4_window7_224    # recomendado para empezar con Swin
--backbone swin_small_patch4_window7_224   # más capacidad
--backbone swin_base_patch4_window7_224    # máxima capacidad
```

> **Nota:** El número `224` es nombre viejo, se puede usar cualquier resolución de entrada

### Otros backbones via timm

Cualquier backbone de timm que soporte `features_only=True` funciona directamente:

```bash
--backbone convnext_tiny
--backbone convnext_base
--backbone efficientnet_b3
```

---

## 3. Elegir el formato de imagen

| Situación | Formato | Argumento |
|-----------|---------|-----------|
| Imágenes PNG estándar, 8 bits | PNG | `--format png` |
| Imágenes radiográficas o de alta dinámica, 16 bits | TIFF | `--format tiff` |

El backbone no distingue el formato — ambos se normalizan a float32 en [0, 1] antes de entrar al modelo. La elección solo afecta a cómo se leen los archivos del disco.

---

## 4. Configurar el entrenamiento

La configuración base está en `experiments/baseline.py`. Los parámetros más importantes:

| Parámetro | Default | Cuándo cambiarlo |
|-----------|---------|-----------------|
| `image_size` | (512, 512) | Si los parches son de otro tamaño |
| `k_shot` | 1 | Si se quieren más imágenes de soporte por episodio |
| `epochs` | 100 | Aumentar si el modelo no converge |
| `learning_rate` | 1e-4 | Reducir si el loss oscila mucho |
| `batch_size` | 4 | Reducir si hay OOM en GPU |
| `dropout_rate` | 0.15 | Aumentar si hay overfitting |
| `augment_support` | True | Desactivar para debugging |
| `augment_query` | True | Desactivar para debugging |

Para cambiar `image_size` u otros parámetros que no tienen argumento CLI, editar directamente `experiments/baseline.py`.

---

## 5. Lanzar el entrenamiento

### Comando básico

```bash
python train.py --format png --backbone resnet34 --data data/
```

### Comando completo con todas las opciones

```bash
python train.py \
  --format png \
  --backbone swin_tiny_patch4_window7_224 \
  --data data/ \
  --epochs 100 \
  --lr 1e-4 \
  --batch_size 4 \
  --k_shot 1 \
  --device cuda \
  --workers 4
```

### Congelar capas del encoder

Útil cuando el dataset es muy pequeño y se quiere evitar que el backbone olvide los pesos preentrenados (a mi me funcionó en dominio de radiografías, congelar 'layer1' del resnet50):

```bash
python train.py --frozen_layers layer1,layer2
```

---

## 6. Monitorear el entrenamiento

El trainer imprime por epoch:

```
Epoch 10/100 | train_loss 0.4231 | train_iou 0.5123 | train_dice 0.6201 | 42.3s
              ||  val_loss 0.4891 | val_iou 0.4876 | val_dice 0.5943  ← best
```

---

## 7. El checkpoint

Cada vez que `val_iou` mejora, se guarda en `checkpoints/baseline/`:

- `best_model.pt` — pesos del modelo
- `best_model.json` — configuración completa del experimento

El JSON documenta todo lo necesario para reproducir o continuar el experimento:

```json
{
  "experiment_name": "baseline_resnet50_1shot",
  "encoder": {
    "backbone": "swin_tiny_patch4_window7_224",
    "pretrained": true,
    "img_size": 256
  },
  "dataset": {
    "image_size": [256, 256],
    "image_format": "png",
    "k_shot": 1
  },
  "training": {
    "epochs": 100,
    "learning_rate": 0.0001
  }
}
```

> **Importante:** Para usar un checkpoint en inferencia, el `image_size` del JSON debe coincidir con el `--patch_size` de `infer.py`.

---

## 8. Inspección visual de predicciones

Antes de pasar a inferencia sobre imágenes completas, verificar visualmente que el modelo predice correctamente sobre el val set:

```bash
python visualize_predictions.py \
  --checkpoint checkpoints/baseline/best_model.pt \
  --data data/ \
  --n_episodes 8 \
  --threshold 0.5 \
  --format png \
  --output results/viz.png
```

---

## 9. Inferencia sobre imágenes completas

Una vez validado el modelo, aplicarlo sobre imágenes completas (sin parchear previamente):

```bash
python infer.py \
  --support_img imagenes_inferencia/support_img.png \
  --support_mask imagenes_inferencia/support_mask.png \
  --query_img imagenes_inferencia/query_img.png \
  --checkpoint checkpoints/baseline/best_model.pt \
  --output results/ \
  --threshold 0.5
```

El script divide la imagen query en parches del tamaño usado en entrenamiento, infiere sobre cada uno y reconstruye la máscara completa.

**Salidas en `results/`:**
- `query.png` — imagen query original
- `pred_mask.png` — máscara de defectos predicha
- `overlay.png` — defectos en rojo sobre la imagen query

### El support para inferencia

El support debe ser un parche representativo del defecto que se quiere detectar — idealmente extraído manualmente de una imagen real con un defecto claramente visible. No tiene que ser del mismo dataset de entrenamiento.

---

