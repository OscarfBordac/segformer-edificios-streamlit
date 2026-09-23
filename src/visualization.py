"""
Visualización de resultados: creación de overlays semitransparentes y mapas de calor.
"""

from typing import Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt


def create_overlay(
    rgb_uint8: np.ndarray,
    mask: np.ndarray,
    color: Union[Tuple[int, int, int], List[int]] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Superpone la máscara de construcciones sobre la imagen original RGB
    utilizando un color y nivel de transparencia definidos.
    """
    overlay = rgb_uint8.copy()
    building_mask = mask.astype(bool)
    building_color = np.array(color, dtype=np.uint8)

    if building_mask.any():
        overlay[building_mask] = (
            (1.0 - alpha) * overlay[building_mask] + alpha * building_color
        ).astype(np.uint8)

    return overlay


def create_heatmap_figure(
    probability: np.ndarray,
    colormap: str = "jet",
    figsize: Tuple[int, int] = (7, 7),
) -> plt.Figure:
    """
    Genera una figura Matplotlib con el mapa de calor de probabilidad continuo [0.0, 1.0].
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(probability, cmap=colormap, vmin=0.0, vmax=1.0)
    ax.set_title("Probabilidad de Construcción", fontsize=12, fontweight="bold")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probabilidad", fontsize=10)

    fig.tight_layout()
    return fig
