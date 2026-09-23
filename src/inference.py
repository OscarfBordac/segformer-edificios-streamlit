"""
Lógica de inferencia del modelo SegFormer:
- Preprocesamiento radiométrico y normalización ImageNet.
- Generación de ventanas espaciales (tiling + stride).
- Test-Time Augmentation (TTA).
- Reconstrucción de probabilidades ponderadas y umbralización.
"""

from typing import List, Tuple, Callable, Optional, Dict, Any
import numpy as np
import torch


def stretch_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """
    Conserva la radiometría para imágenes uint8.
    Para imágenes de 16 bits o coma flotante, aplica un estiramiento
    de contraste por percentiles 2-98% adaptado a teledetección.
    """
    rgb = np.asarray(rgb)

    if rgb.dtype == np.uint8:
        return rgb

    rgb_float = rgb.astype(np.float32)
    out = np.zeros_like(rgb_float, dtype=np.float32)

    for b in range(3):
        lo, hi = np.percentile(rgb_float[:, :, b], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[:, :, b] = np.clip((rgb_float[:, :, b] - lo) / (hi - lo) * 255.0, 0, 255)

    return out.astype(np.uint8)


def preprocess_tile(tile_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Aplica la normalización idéntica al entrenamiento:
    Escalado a [0, 1] y normalización ImageNet estándar (MiT-B2).
    """
    x = tile_rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # (C, H, W)

    tensor = torch.from_numpy(x).unsqueeze(0).float().to(device)
    return tensor


def spatial_windows(
    height: int,
    width: int,
    tile_size: int = 256,
    stride: int = 192,
) -> List[Tuple[int, int]]:
    """
    Calcula las coordenadas de las ventanas espaciales asegurando
    cobertura completa de la imagen, incluidos los bordes inferiores y derechos.
    """
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))

    if not ys or ys[-1] + tile_size < height:
        ys.append(max(height - tile_size, 0))

    if not xs or xs[-1] + tile_size < width:
        xs.append(max(width - tile_size, 0))

    # Eliminar duplicados preservando orden
    unique_windows = []
    seen = set()
    for y in ys:
        for x in xs:
            if (y, x) not in seen:
                seen.add((y, x))
                unique_windows.append((y, x))

    return unique_windows


def predict_tta(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """
    Ejecuta Test-Time Augmentation (TTA) con 4 transformaciones geométricas:
    1. Imagen original (identidad)
    2. Flip horizontal
    3. Flip vertical
    4. Flip horizontal + vertical

    Invierte cada transformación sobre los logits predichos y promedia los resultados.
    """
    variants = [
        (tensor, lambda z: z),
        (torch.flip(tensor, dims=[3]), lambda z: torch.flip(z, dims=[3])),
        (torch.flip(tensor, dims=[2]), lambda z: torch.flip(z, dims=[2])),
        (torch.flip(tensor, dims=[2, 3]), lambda z: torch.flip(z, dims=[2, 3])),
    ]

    accumulated = None

    with torch.inference_mode():
        for x, undo_func in variants:
            logits = model(x)
            logits = undo_func(logits)

            if accumulated is None:
                accumulated = logits
            else:
                accumulated = accumulated + logits

    return accumulated / float(len(variants))


def run_tiled_inference(
    model: torch.nn.Module,
    rgb_input: np.ndarray,
    config: dict,
    device: torch.device,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de inferencia con ventaneo deslizante,
    ponderación por solapamiento y umbralización óptima.
    """
    rgb_uint8 = stretch_to_uint8(rgb_input)
    height, width = rgb_uint8.shape[:2]

    tile_size = int(config.get("input_size", 256))
    stride = int(config.get("stride", 192))
    threshold = float(config.get("threshold", 0.35))
    use_tta = bool(config.get("tta", True))

    windows = spatial_windows(height, width, tile_size=tile_size, stride=stride)
    total_tiles = len(windows)

    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for i, (y, x) in enumerate(windows, start=1):
        y2 = min(y + tile_size, height)
        x2 = min(x + tile_size, width)

        tile = rgb_uint8[y:y2, x:x2]

        # Relleno seguro si el tile se ubica en los bordes y mide menos de tile_size
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        padded[:tile.shape[0], :tile.shape[1]] = tile

        tensor = preprocess_tile(padded, device)

        if use_tta:
            logits = predict_tta(model, tensor)
        else:
            with torch.inference_mode():
                logits = model(tensor)

        # Probabilidad de la clase 'building' (índice 1)
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

        # Recortar al tamaño real del parche (sin padding)
        prob_valid = prob[:tile.shape[0], :tile.shape[1]]

        probability_sum[y:y2, x:x2] += prob_valid
        weight_sum[y:y2, x:x2] += 1.0

        if progress_callback:
            progress_callback(i / total_tiles, i, total_tiles)

    # Promedio ponderado de zonas con solapamiento
    probability = probability_sum / np.maximum(weight_sum, 1e-7)

    # Generación de la máscara binaria aplicando el threshold calibrado
    mask = (probability >= threshold).astype(np.uint8)

    return {
        "rgb_uint8": rgb_uint8,
        "probability": probability,
        "mask": mask,
        "tiles_count": total_tiles,
        "threshold": threshold,
    }
