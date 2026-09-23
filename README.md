# 🏢 Detección y Segmentación de Construcciones con SegFormer (MiT-B2)

Aplicación web interactiva basada en **Streamlit** y **PyTorch** para la detección y segmentación semántica automática de huellas de edificaciones (*building footprints*) a partir de imágenes satelitales y aéreas en formato **GeoTIFF georreferenciado**.

El sistema procesa imágenes de cualquier resolución mediante inferencia por ventanas deslizantes (*tiling* con solapamiento), aplica *Test-Time Augmentation* (TTA) y exporta los resultados conservando con absoluta exactitud el **CRS, resolución espacial, dimensiones y matriz de transformación afín** de la imagen original.

---

## 📑 Tabla de Contenidos
1. [Descripción del Proyecto](#1-qué-hace-el-proyecto)
2. [Arquitectura y Metodología del Modelo](#2-modelo-utilizado)
3. [Dataset y Entrenamiento](#3-dataset-utilizado-en-entrenamiento)
4. [Instalación Local](#4-cómo-instalarlo-localmente)
5. [Ejecución de la Aplicación](#5-cómo-ejecutar-streamlit)
6. [Subida a GitHub y Gestión del Modelo](#6-cómo-subirlo-a-github)
7. [Despliegue en Streamlit Community Cloud](#7-cómo-desplegarlo-en-streamlit-community-cloud)
8. [Guía de Uso de la Aplicación](#8-cómo-utilizar-la-aplicación)
9. [Formatos y Requisitos de Entrada](#9-qué-formatos-de-entrada-acepta)
10. [Resultados y Productos Generados](#10-qué-resultados-genera)

---

## 1. ¿Qué hace el proyecto?
Esta herramienta permite a analistas SIG, investigadores y profesionales de teledetección cargar ortofotos o imágenes satelitales en formato GeoTIFF y obtener de manera inmediata:
* Detección precisa de huellas de edificios y construcciones.
* Generación de mapas de calor continuos de probabilidad.
* Superposición (*overlay*) visual de las construcciones detectadas.
* Descarga directa de archivos GeoTIFF compatibles con cualquier software GIS (**QGIS**, **ArcGIS**, etc.) listos para análisis espacial y cálculo de métricas urbanísticas.

---

## 2. Modelo Utilizado
* **Arquitectura:** [SegFormer](https://arxiv.org/abs/2105.15203) (Semantic Segmentation with Transformers).
* **Encoder / Backbone:** `MiT-B2` (Mix Transformer B2, preentrenado en ImageNet).
* **Clases (2):**
  - `0`: Background (Fondo / Terreno)
  - `1`: Building (Construcción / Edificación)
* **Entrada del modelo:** Mosaicos (*tiles*) de $256 \times 256$ píxeles.
* **Pérdida combinada:**
  $$\mathcal{L} = 0.5 \cdot \text{DiceLoss} + 0.5 \cdot \text{FocalLoss}$$
* **Optimizador y Planificador:** AdamW ($\text{LR} = 1\times 10^{-4}$, $\text{Weight Decay} = 1\times 10^{-4}$) con `CosineAnnealingLR`.
* **Inferencia por Ventanas (Tiling):** Tamaño de ventana $256\text{ px}$, paso (*stride*) de $192\text{ px}$, solapamiento (*overlap*) de $64\text{ px}$.
* **Test-Time Augmentation (TTA):** Promedio de 4 variantes geométricas (original, flip horizontal, flip vertical y flip doble) con inversión de transformaciones en los logits.
* **Umbral de Decisión (Threshold):** `0.35`, calibrado sobre el conjunto de validación para maximizar el IoU (Jaccard Index) y F1-Score.

---

## 3. Dataset Utilizado en Entrenamiento
El modelo fue entrenado y evaluado sobre imágenes aéreas y satelitales de alta resolución espacial con anotaciones de huellas de edificaciones:
* **Resolución espacial típica:** $\sim 0.3 \text{ m}$ a $0.5 \text{ m}$ por píxel.
* **Preparación:** Partición en parches de $256 \times 256\text{ px}$ con data augmentation (rotaciones, flips, variaciones radiométricas de brillo y contraste).
* **Balance de clases:** Optimizado mediante la combinación de Dice Loss y Focal Loss para mitigar el desbalance entre píxeles de fondo y construcciones.

---

## 4. Cómo Instalarlo Localmente

### Requisitos Previos
* **Python 3.10 o 3.11** recomendado.
* Soporte opcional para GPU NVIDIA con CUDA (si no se dispone de GPU, el código corre automáticamente en CPU).

### Paso a Paso
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   cd TU_REPOSITORIO
   ```

2. Crear y activar un entorno virtual:
   * **Windows:**
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Instalar las dependencias:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Colocar los pesos del modelo:
   Asegúrate de que el archivo del checkpoint `segformer_mit_b2_tileado_edificios_best.pth` esté ubicado en la carpeta `model/`:
   ```text
   model/segformer_mit_b2_tileado_edificios_best.pth
   ```

---

## 5. Cómo Ejecutar Streamlit

Con el entorno virtual activado, ejecuta en la terminal:
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador web en la dirección local:
`http://localhost:8501`

---

## 6. Cómo Subirlo a GitHub

### ⚠️ Importante sobre el tamaño del modelo (`.pth`)
El archivo de pesos pesa **~94.4 MB** (`99,032,455` bytes). GitHub tiene un límite estricto de **100 MB** por archivo y emite advertencias a partir de **50 MB**. Para evitar problemas de subida o rechazos en el push, se recomiendan dos estrategias:

#### Opción A: Usar Git LFS (Git Large File Storage) — *Recomendada para Git*
1. Instalar Git LFS en tu equipo: [https://git-lfs.com](https://git-lfs.com)
2. Inicializar Git LFS en el repositorio:
   ```bash
   git lfs install
   git lfs track "model/*.pth"
   git add .gitattributes
   ```
3. Realizar el commit y push habitual:
   ```bash
   git add .
   git commit -m "feat: Detección de construcciones con SegFormer y Streamlit"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   git push -u origin main
   ```

#### Opción B: Subir a GitHub Releases / Hugging Face (Descarga Automática)
Si prefieres no usar Git LFS:
1. En `.gitignore`, descomenta la línea `model/*.pth` para no subir el archivo pesado al control de versiones.
2. Sube el código fuente a GitHub:
   ```bash
   git add .
   git commit -m "feat: Inicialización de la aplicación web de segmentación"
   git push -u origin main
   ```
3. En la página de tu repositorio en GitHub, ve a **Releases** > **Create a new release** y adjunta el archivo `segformer_mit_b2_tileado_edificios_best.pth`.
4. Copia el enlace de descarga directa del asset y pégalo en el campo `"model_download_url"` dentro de `config.json`. La aplicación descargará el modelo automáticamente si no lo encuentra localmente.

---

## 7. Cómo Desplegarlo en Streamlit Community Cloud

1. Ingresa a [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con tu cuenta de GitHub.
2. Haz clic en el botón **New app**.
3. Configura los campos del formulario:
   * **Repository:** Selecciona tu repositorio (ej. `TU_USUARIO/segformer_streamlit_deployment`).
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. En **Advanced settings**:
   * Asegúrate de seleccionar **Python 3.10** o **3.11**.
5. Haz clic en **Deploy!**. Streamlit Cloud instalará las librerías listadas en `requirements.txt` y levantará la aplicación web con una URL pública gratuita.

---

## 8. Cómo Utilizar la Aplicación

1. **Subir imagen:** Arrastra o selecciona tu archivo `.tif` o `.tiff` en el cargador superior.
2. **Inspección espacial:** Observa los metadatos automáticos: dimensiones en píxeles, número de bandas, CRS y resolución.
3. **Ajuste del umbral (opcional):** En la barra lateral puedes modificar el umbral de probabilidad (por defecto `0.35`).
4. **Ejecutar detección:** Pulsa el botón **🚀 Ejecutar Detección de Construcciones**.
5. **Revisar visualizaciones:** Inspecciona el panel cuádruple comparando la imagen original, el overlay en rojo, la máscara binaria y el mapa de calor de probabilidad.
6. **Descargar resultados:** Utiliza los botones de descarga situados al final de la página para obtener los GeoTIFFs georreferenciados.

---

## 9. ¿Qué Formatos de Entrada Acepta?

* **Formato:** Archivos raster **GeoTIFF** (`.tif`, `.tiff`).
* **Georreferenciación requerida:** El archivo **debe** incluir un CRS válido (ej. EPSG:4326, EPSG:32618) y una matriz de transformación espacial afín.
* **Canales / Bandas:**
  * **3 o más bandas:** El sistema toma automáticamente las tres primeras bandas y las trata como RGB.
  * **1 banda (monocanal):** La banda se replica de forma segura en los canales R, G y B.
* **Profundidad de bits:**
  * Soporta imágenes estándar de 8 bits (`uint8`).
  * Para imágenes de 16 bits (`uint16`) o coma flotante (`float32`), se aplica automáticamente un estiramiento de contraste por percentiles (2% - 98%) para alimentar el modelo con la radiometría adecuada.
* **Dimensiones:** No hay restricción de tamaño. Si la imagen es mayor a $256 \times 256$ o sus dimensiones no son múltiplos exactos de 256, el algoritmo de ventaneo deslizante cubre todos los bordes sin dejar zonas sin procesar.

---

## 10. ¿Qué Resultados Genera?

La aplicación genera 3 productos cartográficos en formato GeoTIFF con compresión `deflate`, manteniendo exactamente el **mismo CRS, resolución, dimensiones y georreferenciación** de la imagen original:

| Archivo | Tipo de Dato | Bandas | Descripción |
| :--- | :---: | :---: | :--- |
| `mask_buildings.tif` | `uint8` | 1 | Máscara binaria clasificada: `0 = Background`, `1 = Building`. |
| `building_probability.tif` | `float32` | 1 | Valores continuos de probabilidad calculados por la red ($0.0 \le p \le 1.0$). |
| `building_overlay.tif` | `uint8` | 3 | Imagen RGB original con las construcciones resaltadas en color rojo semitransparente ($\alpha = 0.35$). |

Todos los archivos generados están listos para ser importados directamente en cualquier SIG sin necesidad de reproyección o ajuste manual.
