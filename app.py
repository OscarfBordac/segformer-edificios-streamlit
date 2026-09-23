"""
Aplicación Web Streamlit: Detección y Segmentación de Construcciones con SegFormer (MiT-B2).
"""

import json
from pathlib import Path
import streamlit as st
import torch
import numpy as np

from src.model import load_segformer_model, resolve_model_path
from src.geotiff import (
    read_geotiff_data,
    export_binary_mask,
    export_building_probability,
    export_building_overlay,
    verify_geotiff_integrity,
)
from src.inference import run_tiled_inference
from src.visualization import create_overlay, create_heatmap_figure


# ============================================================
# CONFIGURACIÓN GENERAL Y DEL SISTEMA
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ROOT_CONFIG_PATH = APP_DIR / "config.json"
MODEL_CONFIG_PATH = APP_DIR / "model" / "config.json"


@st.cache_data
def load_application_config() -> dict:
    """Carga el archivo de configuración unificado con fallback."""
    config_file = ROOT_CONFIG_PATH if ROOT_CONFIG_PATH.exists() else MODEL_CONFIG_PATH
    if not config_file.exists():
        raise FileNotFoundError(
            f"No se localizó el archivo de configuración en:\n{ROOT_CONFIG_PATH}"
        )
    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("threshold") is None:
        config["threshold"] = 0.35
    return config


CONFIG = load_application_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# GESTIÓN DEL MODELO EN MEMORIA (CACHE)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Carga y mantiene en memoria el modelo SegFormer."""
    return load_segformer_model(CONFIG, device=DEVICE)


# ============================================================
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Detección de Construcciones con SegFormer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Encabezado principal
st.title("Detección de Construcciones con SegFormer")
st.subheader("Segmentación automática de huellas de construcciones a partir de imágenes satelitales")
st.markdown("---")

# ============================================================
# BARRA LATERAL (INFORMACIÓN TÉCNICA Y PARÁMETROS)
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuración del Modelo")

    st.markdown(f"**Arquitectura:** `{CONFIG.get('architecture', 'SegFormer')}`")
    st.markdown(f"**Encoder:** `{CONFIG.get('encoder', 'MiT-B2')}`")
    st.markdown(f"**Clases:** `{', '.join(CONFIG.get('classes', ['background', 'building']))}`")
    st.markdown(f"**Tamaño de Tile:** `{CONFIG.get('input_size', 256)} × {CONFIG.get('input_size', 256)} px`")
    st.markdown(f"**Paso (Stride):** `{CONFIG.get('stride', 192)} px`")
    st.markdown(f"**Solapamiento (Overlap):** `{int(CONFIG.get('input_size', 256)) - int(CONFIG.get('stride', 192))} px`")
    st.markdown(f"**TTA (Test-Time Aug.):** `{'Activado (4 pasadas)' if CONFIG.get('tta', True) else 'Desactivado'}`")

    device_str = "🟢 GPU CUDA" if torch.cuda.is_available() else "🟠 CPU"
    st.markdown(f"**Dispositivo de Cómputo:** {device_str}")

    st.markdown("---")
    st.subheader("🎛️ Umbral de Detección")
    default_thresh = float(CONFIG.get("threshold", 0.35))
    selected_threshold = st.slider(
        "Threshold de probabilidad:",
        min_value=0.05,
        max_value=0.95,
        value=default_thresh,
        step=0.05,
        help="Valor de probabilidad a partir del cual un píxel es clasificado como construcción. Calibrado en validación: 0.35",
    )

    # Estado del checkpoint
    st.markdown("---")
    resolved_pth = resolve_model_path(CONFIG.get("model_path"))
    if resolved_pth.is_file():
        file_size_mb = resolved_pth.stat().st_size / (1024 * 1024)
        st.success(f"Checkpoint cargado: `{resolved_pth.name}` ({file_size_mb:.1f} MB)")
    else:
        st.warning("⚠️ Checkpoint no encontrado localmente. Se verificará en inferencia.")

# ============================================================
# CUERPO PRINCIPAL DE LA APLICACIÓN
# ============================================================

st.markdown(
    """
    Sube una imagen satelital o aérea en formato **GeoTIFF georreferenciado** (`.tif` o `.tiff`).
    El sistema dividirá automáticamente la imagen en ventanas con solapamiento, aplicará el modelo 
    **SegFormer MiT-B2** con TTA y exportará resultados que conservan con exactitud:
    **CRS, matriz de transformación afín, resolución espacial y extensión geográfica.**
    """
)

uploaded_file = st.file_uploader(
    "Seleccionar archivo GeoTIFF",
    type=["tif", "tiff"],
    help="Formatos admitidos: GeoTIFF georreferenciados (monocanal o multicanal RGB).",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    # 1. Inspección preliminar de metadatos geoespaciales
    try:
        geo_info = read_geotiff_data(file_bytes)

        st.success("✅ GeoTIFF válido y georreferenciado.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dimensiones", f"{geo_info['width']} × {geo_info['height']} px")
        m2.metric("Bandas", str(geo_info["count"]))
        m3.metric("CRS", str(geo_info["crs"].to_string() if hasattr(geo_info["crs"], "to_string") else geo_info["crs"]))
        m4.metric("Resolución de píxel", f"{geo_info['resolution'][0]:.2f}, {geo_info['resolution'][1]:.2f}")

        with st.expander("📍 Ver extensión espacial (Bounds) y perfil"):
            st.write(f"**Bounding Box:** `{geo_info['bounds']}`")
            st.write(f"**Matriz de Transformación Afín:** `{geo_info['transform']}`")

        # Botón para disparar la inferencia
        run_btn = st.button("🚀 Ejecutar Detección de Construcciones", type="primary", use_container_width=True)

        if run_btn:
            # Sobrescribir threshold con el seleccionado por el usuario en la sesión
            inference_config = CONFIG.copy()
            inference_config["threshold"] = selected_threshold

            # Carga del modelo
            try:
                with st.spinner("Cargando modelo SegFormer en memoria..."):
                    model = get_cached_model()
            except Exception as e:
                st.error(f"Error al inicializar el modelo: {e}")
                st.stop()

            # Inferencia con barra de progreso
            progress_bar = st.progress(0.0, text="Iniciando segmentación...")
            status_text = st.empty()

            def update_progress(ratio, current, total):
                progress_bar.progress(ratio, text=f"Procesando tile {current}/{total}...")

            try:
                with st.spinner("Procesando imagen con ventana deslizante y TTA..."):
                    inf_result = run_tiled_inference(
                        model=model,
                        rgb_input=geo_info["rgb"],
                        config=inference_config,
                        device=DEVICE,
                        progress_callback=update_progress,
                    )

                progress_bar.progress(1.0, text="Inferencia finalizada. Generando productos geoespaciales...")
                status_text.empty()

                rgb_uint8 = inf_result["rgb_uint8"]
                probability = inf_result["probability"]
                mask = inf_result["mask"]

                # Generación de visualizaciones
                overlay = create_overlay(rgb_uint8, mask, color=(255, 0, 0), alpha=0.35)
                heatmap_fig = create_heatmap_figure(probability)

                # Generación de bytes GeoTIFF georreferenciados en memoria
                with st.spinner("Serializando GeoTIFFs con metadatos espaciales intactos..."):
                    mask_bytes = export_binary_mask(mask, geo_info["profile"])
                    prob_bytes = export_building_probability(probability, geo_info["profile"])
                    overlay_bytes = export_building_overlay(overlay, geo_info["profile"])

                    # Validación de integridad geoespacial
                    verify_geotiff_integrity(mask_bytes, geo_info)
                    verify_geotiff_integrity(prob_bytes, geo_info)
                    verify_geotiff_integrity(overlay_bytes, geo_info)

                progress_bar.empty()
                st.success("🎉 Detección completada exitosamente. Integridad geoespacial verificada al 100%.")

                # ============================================================
                # RESULTADOS VISUALES (PANEL 2x2)
                # ============================================================
                st.subheader("🖼️ Resultados Visuales")

                row1_col1, row1_col2 = st.columns(2)
                with row1_col1:
                    st.image(rgb_uint8, caption="1. Imagen Original (RGB)", use_container_width=True)
                with row1_col2:
                    st.image(overlay, caption="2. Overlay de Construcciones (Rojo)", use_container_width=True)

                row2_col1, row2_col2 = st.columns(2)
                with row2_col1:
                    st.image(mask * 255, caption=f"3. Máscara Binaria (Threshold: {selected_threshold:.2f})", use_container_width=True)
                with row2_col2:
                    st.pyplot(heatmap_fig, clear_figure=True)

                # ============================================================
                # MÉTRICAS Y ESTADÍSTICAS DEL ANÁLISIS
                # ============================================================
                st.subheader("📊 Resumen Cuantitativo")
                total_pixels = mask.size
                building_pixels = int(np.sum(mask))
                coverage_pct = (building_pixels / total_pixels) * 100.0

                s1, s2, s3 = st.columns(3)
                s1.metric("Tiles Procesados", inf_result["tiles_count"])
                s2.metric("Píxeles de Construcción", f"{building_pixels:,}")
                s3.metric("Cobertura Estimada", f"{coverage_pct:.2f}%")

                # ============================================================
                # DESCARGA DE PRODUCTOS GEORREFERENCIADOS
                # ============================================================
                st.subheader("💾 Descarga de Resultados (GeoTIFF)")
                st.markdown(
                    "Todos los archivos descargables conservan **el mismo CRS, resolución, dimensiones "
                    "y coordenadas del raster original**, listos para ser abiertos en QGIS o ArcGIS."
                )

                d1, d2, d3 = st.columns(3)
                with d1:
                    st.download_button(
                        label="📥 Descargar mask_buildings.tif",
                        data=mask_bytes,
                        file_name="mask_buildings.tif",
                        mime="image/tiff",
                        use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        label="📥 Descargar building_probability.tif",
                        data=prob_bytes,
                        file_name="building_probability.tif",
                        mime="image/tiff",
                        use_container_width=True,
                    )
                with d3:
                    st.download_button(
                        label="📥 Descargar building_overlay.tif",
                        data=overlay_bytes,
                        file_name="building_overlay.tif",
                        mime="image/tiff",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Ocurrió un error durante la inferencia: {e}")

    except Exception as e:
        st.error(f"Error al procesar el archivo GeoTIFF: {e}")
