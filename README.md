# **DeepWind: Detección de Defectos en Aerogeneradores con Deep Learning**

🎯 ## Descripción del Proyecto
DeepWind es una solución basada en Deep Learning (YOLO) para la detección automática de defectos en aerogeneradores.

🛠️ ## Requisitos
```
kaggle
ultralytics
gradio
Pillow
opencv-python
```

📊 ## Dataset
El dataset se descarga automáticamente usando la API de Kaggle. Para configurarlo:

1. Asegúrate de tener credenciales de Kaggle `(~/.kaggle/kaggle.json)`
2. Ejecuta el script de descarga:
   ```
   python download_data.py
   ```
3. El script organizará los datos en el siguiente formato:
   `
   yolo_dataset/
   ├── train/
   ├── test/
   ├── valid/
   └── data.yaml
   `
4. 🚀 ## Entrenamiento
### **Opción 1: Entrenamiento Local**
Ejecuta el script de entrenamiento:
```
from train import train_model

results = train_model(
    data_yaml="yolo_dataset/data.yaml",
    epochs=50,
    batch_size=32,
    optimizer="AdamW"
)
```

### **Opción 2: Google Colab**
Abre `deepwind_colab_training.ipynb`
 en Google Colab. [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/ErikSarriegui/DeepWind/blob/main/deepwind_colab_training.ipynb)
