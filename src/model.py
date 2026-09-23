"""
Gestión y carga del modelo SegFormer y sus pesos entrenados.
"""

from pathlib import Path
import urllib.request
import torch
import segmentation_models_pytorch as smp


def resolve_model_path(configured_path: str = None) -> Path:
    """
    Busca de forma flexible la ruta del archivo checkpoint .pth.
    Primero verifica la ruta configurada, luego busca en la carpeta model/
    y en el directorio raíz.
    """
    base_dir = Path(__file__).resolve().parent.parent

    # 1. Comprobar ruta directa o relativa a la raíz del proyecto
    if configured_path:
        direct_path = Path(configured_path)
        if direct_path.is_file():
            return direct_path
        relative_path = base_dir / configured_path
        if relative_path.is_file():
            return relative_path

    # 2. Comprobar nombre por defecto en carpeta model/
    default_in_model = base_dir / "model" / "segformer_mit_b2_tileado_edificios_best.pth"
    if default_in_model.is_file():
        return default_in_model

    # 3. Buscar cualquier archivo .pth en la carpeta model/
    model_dir = base_dir / "model"
    if model_dir.exists():
        pth_files = list(model_dir.glob("*.pth"))
        if pth_files:
            return pth_files[0]

    # 4. Buscar en el directorio raíz
    root_pth = list(base_dir.glob("*.pth"))
    if root_pth:
        return root_pth[0]

    return default_in_model


def download_model_if_needed(target_path: Path, download_url: str = None) -> bool:
    """
    Si el archivo .pth no existe y se ha proporcionado una URL de descarga válida
    (por ejemplo, un enlace directo de Hugging Face o GitHub Releases),
    descarga el checkpoint automáticamente de forma segura.
    """
    if target_path.is_file() and target_path.stat().st_size > 1000:
        return True

    if not download_url or not download_url.strip():
        return False

    download_url = download_url.strip()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".tmp")
    print(f"Descargando checkpoint del modelo desde: {download_url}...")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StreamlitApp"}

    try:
        import requests
        with requests.get(download_url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            with open(temp_target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if temp_target.is_file() and temp_target.stat().st_size > 1000:
            temp_target.replace(target_path)
            print(f"Checkpoint descargado exitosamente en: {target_path}")
            return True
    except Exception as e_req:
        print(f"Intento con requests falló ({e_req}). Probando con urllib...")
        try:
            req = urllib.request.Request(download_url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response, open(temp_target, "wb") as out_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
            if temp_target.is_file() and temp_target.stat().st_size > 1000:
                temp_target.replace(target_path)
                print(f"Checkpoint descargado exitosamente en: {target_path}")
                return True
        except Exception as e_url:
            print(f"Error definitivo al descargar checkpoint: {e_url}")
            if temp_target.exists():
                temp_target.unlink()
            return False

    return False


def load_segformer_model(config: dict, device: torch.device = None) -> torch.nn.Module:
    """
    Carga el modelo SegFormer con encoder MiT-B2 y carga los pesos del checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = resolve_model_path(config.get("model_path"))

    if not model_path.is_file():
        # Intentar descarga si hay URL disponible
        download_url = config.get("model_download_url", "").strip()
        downloaded = download_model_if_needed(model_path, download_url)
        if not downloaded or not model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el checkpoint del modelo en:\n{model_path}\n\n"
                "Asegúrate de colocar el archivo 'segformer_mit_b2_tileado_edificios_best.pth' "
                "dentro del directorio 'model/' o configurar una URL de descarga en 'config.json'."
            )

    encoder_name = config.get("encoder", "mit_b2")
    num_classes = config.get("num_classes", 2)

    # Instanciación de SegFormer según el entrenamiento
    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=None,
        classes=num_classes,
        activation=None,
    )

    # Carga segura y compatible del diccionario de pesos
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    # Manejar si el state_dict viene empaquetado en un diccionario
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model
