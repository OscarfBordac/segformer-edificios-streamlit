"""
Manejo de archivos GeoTIFF con Rasterio: lectura, validación espacial y exportación.
"""

from io import BytesIO
from typing import Dict, Any, Union
from pathlib import Path
import numpy as np
import rasterio
from rasterio.io import MemoryFile


def read_geotiff_data(file_input: Union[bytes, BytesIO, Path, str]) -> Dict[str, Any]:
    """
    Lee un archivo GeoTIFF preservando íntegramente sus propiedades geoespaciales.
    Acepta bytes, un objeto tipo archivo o una ruta en disco.
    """
    def _extract_from_dataset(src):
        if src.crs is None:
            raise ValueError(
                "El GeoTIFF no cuenta con un CRS (Sistema de Referencia de Coordenadas) asignado. "
                "Por favor, sube un archivo debidamente georreferenciado."
            )

        if src.transform is None:
            raise ValueError(
                "El GeoTIFF no contiene la matriz de transformación espacial afín."
            )

        # Manejo de bandas de entrada
        if src.count >= 3:
            # Seleccionar las primeras 3 bandas como canales R, G, B
            raster = src.read([1, 2, 3])
        elif src.count == 1:
            # Si el raster es monocanal, replicar en 3 canales para compatibilidad con el modelo
            band = src.read(1)
            raster = np.stack([band, band, band], axis=0)
        else:
            raise ValueError(
                f"El raster contiene {src.count} bandas, una cantidad no soportada (se requiere 1 o >=3)."
            )

        # Transponer de (Canales, Alto, Ancho) a (Alto, Ancho, Canales)
        rgb = np.transpose(raster, (1, 2, 0))

        return {
            "rgb": rgb,
            "profile": src.profile.copy(),
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "resolution": src.res,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": src.dtypes,
        }

    if isinstance(file_input, bytes):
        with MemoryFile(file_input) as memfile:
            with memfile.open() as src:
                return _extract_from_dataset(src)
    elif hasattr(file_input, "read"):
        content = file_input.read()
        with MemoryFile(content) as memfile:
            with memfile.open() as src:
                return _extract_from_dataset(src)
    else:
        with rasterio.open(file_input) as src:
            return _extract_from_dataset(src)


def array_to_geotiff_bytes(array: np.ndarray, profile: dict) -> bytes:
    """
    Convierte un arreglo NumPy a bytes de GeoTIFF manteniendo la georreferenciación.
    """
    profile_to_use = profile.copy()
    with MemoryFile() as memfile:
        with memfile.open(**profile_to_use) as dst:
            if profile_to_use.get("count", 1) == 1:
                # Si el array es 2D (Alto, Ancho)
                if array.ndim == 3 and array.shape[2] == 1:
                    array = array[:, :, 0]
                dst.write(array, 1)
            else:
                # Si el array es 3D (Alto, Ancho, Canales) -> pasar a (Canales, Alto, Ancho)
                if array.ndim == 3 and array.shape[2] == profile_to_use.get("count", 3):
                    array_to_write = np.transpose(array, (2, 0, 1))
                else:
                    array_to_write = array
                dst.write(array_to_write)

        return memfile.read()


def export_binary_mask(mask: np.ndarray, base_profile: dict) -> bytes:
    """
    Genera un archivo GeoTIFF para la máscara binaria (0 = background, 1 = building).
    """
    profile = base_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        compress="deflate",
        nodata=0,
        width=mask.shape[1],
        height=mask.shape[0],
    )
    return array_to_geotiff_bytes(mask.astype(np.uint8), profile)


def export_building_probability(probability: np.ndarray, base_profile: dict) -> bytes:
    """
    Genera un archivo GeoTIFF con los valores continuos de probabilidad (float32, 0.0 a 1.0).
    """
    profile = base_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate",
        nodata=-9999.0,
        width=probability.shape[1],
        height=probability.shape[0],
    )
    return array_to_geotiff_bytes(probability.astype(np.float32), profile)


def export_building_overlay(overlay: np.ndarray, base_profile: dict) -> bytes:
    """
    Genera un archivo GeoTIFF RGB con el overlay visual georreferenciado.
    """
    profile = base_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=3,
        compress="deflate",
        nodata=None,
        width=overlay.shape[1],
        height=overlay.shape[0],
    )
    return array_to_geotiff_bytes(overlay.astype(np.uint8), profile)


def verify_geotiff_integrity(geotiff_bytes: bytes, reference_metadata: dict) -> bool:
    """
    Verifica que el GeoTIFF generado en memoria conserve con total exactitud
    el CRS, resolución, dimensiones y transformación afín de referencia.
    """
    with MemoryFile(geotiff_bytes) as memfile:
        with memfile.open() as ds:
            assert ds.crs == reference_metadata["crs"], "El CRS no coincide."
            assert ds.transform == reference_metadata["transform"], "La transformación afín no coincide."
            assert ds.width == reference_metadata["width"], "El ancho no coincide."
            assert ds.height == reference_metadata["height"], "La altura no coincide."
            assert ds.res == reference_metadata["resolution"], "La resolución espacial no coincide."
    return True
