import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Basisverzeichnis des Skripts
BASE_DIR = Path(__file__).resolve().parent

# --- Konfiguration ---
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "fasterrcnn_mobilenet_v3_finetuned.pth"
DEFAULT_IMAGE_DIR = BASE_DIR / "own_images"  # Verzeichnis, in dem sich alle Bilder befinden
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
DEFAULT_OUTPUT_DIR = BASE_DIR / "inference_results"
DEFAULT_THRESHOLD = 0.5
# ---------------------

def load_model(checkpoint_path: Path, device: torch.device, num_classes: int = 2):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def collect_images_from_dir(directory: Path):
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    return sorted([p.resolve() for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def run_inference(model, image_paths, output_dir: Path, device, threshold: float = DEFAULT_THRESHOLD):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir}")
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
            print(f"Saved inference result to {save_path}")
        else:
            print(f"[ERROR] Konnte Datei nicht speichern: {save_path}")


def main(image_list, checkpoint, output, threshold):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model = load_model(Path(checkpoint), device)
    run_inference(model, image_list, Path(output), device, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run face-detection inference")
    parser.add_argument("--images", nargs='+', help="List of image file paths")
    parser.add_argument("--checkpoint", type=str, help="Path to model .pth checkpoint")
    parser.add_argument("--output", type=str, help="Directory for results")
    parser.add_argument("--threshold", type=float, help="Score threshold")
    args = parser.parse_args()

    # Bilderliste aus CLI oder aus default-Verzeichnis
    if args.images:
        image_list = [Path(p) for p in args.images]
    else:
        image_list = collect_images_from_dir(DEFAULT_IMAGE_DIR)

    checkpoint_path = args.checkpoint if args.checkpoint else DEFAULT_CHECKPOINT
    output_dir = args.output if args.output else DEFAULT_OUTPUT_DIR
    threshold = args.threshold if args.threshold else DEFAULT_THRESHOLD

    main(image_list, checkpoint_path, output_dir, threshold)
