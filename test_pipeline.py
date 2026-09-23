"""
Script de pruebas unitarias y de integración para validar el pipeline completo:
1. Verificación de imports y sintaxis.
2. Carga del checkpoint del modelo SegFormer.
3. Inferencia sobre imagen pequeña (< 256x256).
4. Inferencia sobre imagen grande (> 256x256 con dimensiones arbitrarias).
5. Verificación de preservación de CRS, transform, resolución y dimensiones.
6. Validación de archivos GeoTIFF generados.
"""

import sys
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
import torch

# Asegurar importación de src
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.model import load_segformer_model, resolve_model_path
from src.geotiff import (
    read_geotiff_data,
    export_binary_mask,
    export_building_probability,
    export_building_overlay,
    verify_geotiff_integrity,
)
from src.inference import run_tiled_inference, spatial_windows
from src.visualization import create_overlay, create_heatmap_figure


def create_synthetic_geotiff(height: int, width: int, bands: int = 3) -> bytes:
    """Crea en memoria un GeoTIFF georreferenciado sintético para pruebas."""
    transform = from_origin(500000.0, 4500000.0, 0.5, 0.5)
    crs = rasterio.crs.CRS.from_epsg(32630)  # UTM zona 30N

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": bands,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
    }

    # Generar datos sintéticos con patrones
    data = np.random.randint(50, 200, size=(bands, height, width), dtype=np.uint8)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(data)
        return memfile.read()


def test_spatial_windows():
    print("\n--- Test 1: Comprobación de Ventanas Espaciales ---")
    windows = spatial_windows(height=350, width=420, tile_size=256, stride=192)
    print(f"Ventanas generadas para 350x420: {len(windows)}")
    # Asegurar que cubra los bordes
    ys = [y for y, x in windows]
    xs = [x for y, x in windows]
    assert max(ys) + 256 >= 350, "El borde inferior no está cubierto."
    assert max(xs) + 256 >= 420, "El borde derecho no está cubierto."
    print("✓ Ventanas espaciales cubren el 100% del área y los bordes.")


def test_model_loading():
    print("\n--- Test 2: Comprobación de Carga del Modelo ---")
    with open(ROOT_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {device}")

    model_path = resolve_model_path(config.get("model_path"))
    print(f"Ruta del modelo resuelta: {model_path} (Existe: {model_path.is_file()})")
    assert model_path.is_file(), "El archivo .pth no existe en la ruta esperada."

    model = load_segformer_model(config, device=device)
    assert model is not None, "El modelo no se cargó correctamente."
    print("✓ Modelo SegFormer MiT-B2 cargado exitosamente en modo eval.")
    return model, config, device


def test_inference_pipeline(model, config, device):
    print("\n--- Test 3: Inferencia sobre Imagen Pequeña (200x200) ---")
    small_bytes = create_synthetic_geotiff(height=200, width=200, bands=3)
    geo_info_small = read_geotiff_data(small_bytes)
    assert geo_info_small["width"] == 200 and geo_info_small["height"] == 200

    res_small = run_tiled_inference(
        model=model,
        rgb_input=geo_info_small["rgb"],
        config=config,
        device=device,
    )
    assert res_small["mask"].shape == (200, 200)
    assert res_small["probability"].shape == (200, 200)
    print("✓ Inferencia sobre imagen pequeña completada con éxito.")

    print("\n--- Test 4: Inferencia sobre Imagen Mayor a 256x256 (350x420) ---")
    large_bytes = create_synthetic_geotiff(height=350, width=420, bands=3)
    geo_info_large = read_geotiff_data(large_bytes)

    res_large = run_tiled_inference(
        model=model,
        rgb_input=geo_info_large["rgb"],
        config=config,
        device=device,
    )
    assert res_large["mask"].shape == (350, 420)
    assert res_large["probability"].shape == (350, 420)
    assert res_large["tiles_count"] > 1
    print(f"✓ Inferencia sobre imagen de 350x420 completada ({res_large['tiles_count']} tiles procesados).")

    print("\n--- Test 5: Visualización y Overlays ---")
    overlay = create_overlay(res_large["rgb_uint8"], res_large["mask"])
    assert overlay.shape == (350, 420, 3)
    fig = create_heatmap_figure(res_large["probability"])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("✓ Creación de overlay y heatmap de probabilidad exitosa.")

    print("\n--- Test 6: Preservación de Georreferenciación y Serialización GeoTIFF ---")
    mask_bytes = export_binary_mask(res_large["mask"], geo_info_large["profile"])
    prob_bytes = export_building_probability(res_large["probability"], geo_info_large["profile"])
    overlay_bytes = export_building_overlay(overlay, geo_info_large["profile"])

    # Validar integridad espacial
    assert verify_geotiff_integrity(mask_bytes, geo_info_large), "Fallo en integridad de máscara."
    assert verify_geotiff_integrity(prob_bytes, geo_info_large), "Fallo en integridad de probabilidad."
    assert verify_geotiff_integrity(overlay_bytes, geo_info_large), "Fallo en integridad de overlay."
    print("✓ Verificación de integridad geoespacial exitosa (CRS, transform, resolución, dimensiones).")


def main():
    print("==================================================")
    print("INICIANDO SUITE DE PRUEBAS DE DETECCIÓN SEGFORMER")
    print("==================================================")
    test_spatial_windows()
    model, config, device = test_model_loading()
    test_inference_pipeline(model, config, device)
    print("\n==================================================")
    print("¡TODAS LAS PRUEBAS FINALIZARON EXITOSAMENTE! (6/6)")
    print("==================================================")


if __name__ == "__main__":
    main()
