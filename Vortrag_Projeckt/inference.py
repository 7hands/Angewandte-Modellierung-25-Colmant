import os
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToTensor, ToPILImage

def run_inference(model, image_paths, output_dir, device, threshold=0.5):
    """
    Run inference on images and save outputs with bounding boxes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = to_tensor(img).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        # Move tensors to CPU
        boxes = outputs['boxes'].cpu()
        scores = outputs['scores'].cpu()

        # Filter by threshold
        keep = scores >= threshold
        if keep.sum().item() == 0:
            print(f"No detections above threshold {threshold} for {img_path}. Lower threshold or check model.")
            continue

        kept_boxes = boxes[keep].round().int()

        # Prepare image tensor on CPU, uint8
        img_cpu = (img_tensor.cpu() * 255).to(torch.uint8)

        # Draw bounding boxes in red with label 'face'
        img_boxes = draw_bounding_boxes(
            img_cpu,
            kept_boxes,
            labels=["face"] * len(kept_boxes),
            colors=["red"] * len(kept_boxes),
            width=3
        )

        # Save the result
        save_path = output_dir / Path(img_path).name
        to_pil(img_boxes).save(save_path)
        print(f"Saved result with {len(kept_boxes)} boxes to {save_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    # Define your image paths
    custom_images = [
        "Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/istockphoto.jpg",
        "Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/people.jpg",
        "Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/Mona_Lisa.jpg"
    ]

    inference_dir = project_root / "inference_results"

    # Device setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize model
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    num_classes = 2  # background + face
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Load trained weights
    checkpoint_path = project_root / "checkpoints" / "fasterrcnn_mobilenet_v3_finetuned.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Run inference
    run_inference(model, custom_images, inference_dir, device, threshold=0)
