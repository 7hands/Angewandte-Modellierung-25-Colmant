import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Stelle sicher, dass onnx installiert ist
try:
    import onnx  # noqa: F401
except ImportError:
    raise ImportError(
        "Das Modul 'onnx' fehlt. Installiere es bitte per 'pip install onnx' "
        "oder füge es zu deinen Abhängigkeiten hinzu."
    )

# Basisverzeichnis des Skripts
BASE_DIR = Path(__file__).resolve().parent

# --- Konfiguration ---
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "fasterrcnn_mobilenet_v3_finetuned.pth"
DEFAULT_IMAGE_DIR = BASE_DIR / "own_images"  # Verzeichnis mit Eingangs-Bildern
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
DEFAULT_OUTPUT_DIR = BASE_DIR / "inference_results"
DEFAULT_THRESHOLD = 0.5
# ---------------------

def load_model(checkpoint_path: Path, device: torch.device, num_classes: int = 2):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Setze weights_only=True, um Pickle-Angriffe zu vermeiden
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def export_to_onnx(model, export_path: Path, device, input_size=(3, 224, 224)):
    """
    Exportiert das gegebene Modell als ONNX-Datei.
    """
    model.eval()
    dummy_input = torch.randn(1, *input_size, device=device)

    export_path = export_path.resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        opset_version=12,
        input_names=["input"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={
            "input": {0: "batch"},
            "boxes": {0: "batch"},
            "scores": {0: "batch"},
            "labels": {0: "batch"},
        }
    )
    print(f"ONNX-Modell erfolgreich exportiert nach: {export_path}")


def collect_images_from_dir(directory: Path):
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    return sorted([p.resolve() for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def run_inference(model, image_paths, output_dir: Path, device, threshold: float = DEFAULT_THRESHOLD):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Speichere Ergebnisse nach: {output_dir}")
    to_tensor = ToTensor()

    for img_path in image_paths:
        img_path = Path(img_path).resolve()
        if not img_path.is_file():
            print(f"[WARN] Datei nicht gefunden: {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        img_tensor = to_tensor(img).to(device)
        with torch.no_grad():
            outputs = model([img_tensor])[0]

        boxes = outputs['boxes']
        scores = outputs['scores']
        labels = outputs['labels']
        keep = scores >= threshold

        img_boxes = draw_bounding_boxes(
            (img_tensor * 255).to(torch.uint8),
            boxes[keep],
            labels=[str(int(l.item())) for l in labels[keep]],
            width=2
        )

        save_path = output_dir / img_path.name
        from torchvision.transforms import ToPILImage
        ToPILImage()(img_boxes).save(save_path)
        if save_path.is_file():
            print(f"Ergebnis gespeichert: {save_path}")
        else:
            print(f"[ERROR] Konnte nicht speichern: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference und ONNX-Export für Faster R-CNN"
    )
    parser.add_argument(
        "--images", nargs='+', help="Pfad(e) zu Bilddateien"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        help="Pfad zum .pth Checkpoint"
    )
    parser.add_argument(
        "--output", type=str,
        help="Verzeichnis für Ergebnisse oder Pfad für ONNX-Datei"
    )
    parser.add_argument(
        "--threshold", type=float,
        help="Schwellwert für Score-Filter"
    )
    parser.add_argument(
        "--export-onnx", action='store_true',
        help="Modell als ONNX exportieren und beenden"
    )
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Verwende Gerät: {device}")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    model = load_model(checkpoint_path, device)

    # ONNX-Export
    if args.export_onnx:
        export_path = Path(args.output) if args.output else BASE_DIR / "model.onnx"
        export_to_onnx(model, export_path, device)
        return

    # Inferenz-Pfad
    if args.images:
        image_list = [Path(p) for p in args.images]
    else:
        image_list = collect_images_from_dir(DEFAULT_IMAGE_DIR)

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    threshold = args.threshold if args.threshold else DEFAULT_THRESHOLD
    run_inference(model, image_list, output_dir, device, threshold)


if __name__ == "__main__":
    main()