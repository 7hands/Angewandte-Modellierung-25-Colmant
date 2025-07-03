import os
from pathlib import Path
import torch
from PIL import Image
import scipy.io
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes # for he boxes around the faces
from torchvision.transforms import ToTensor # to load the trained model.

def run_inference(model, image_paths, output_dir, device, threshold=0.5):
    """
    Run inference on a list of image files and save visualized outputs.
    """
    
    output_dir = Path(output_dir) # Vortrag_Projeckt\inference_results
    output_dir.mkdir(parents=True, exist_ok=True) # if not there create output directory

    model.eval() # switch mode to eval for inference 
    to_tensor = ToTensor()
    for img_path in image_paths:        
        img = Image.open(img_path).convert("RGB") # open the image in Rgb
        img_tensor = to_tensor(img).to(device) # convert the image to a array like Tensor
        with torch.no_grad():
            outputs = model([img_tensor])[0]
        # outputs
        boxes = outputs['boxes'] 
        scores = outputs['scores']
        labels = outputs['labels']
        keep = scores >= threshold
        # Draw the box in the image
        img_boxes = draw_bounding_boxes(
            (img_tensor * 255).to(torch.uint8),
            boxes[keep],
            labels=[str(int(l.item())) for l in labels[keep]],
            width=2
        )
        # save the new image to the output directory
        save_path = output_dir / Path(img_path).name
        T.ToPILImage()(img_boxes).save(save_path)
        print(f"Saved inference result to {save_path}")
project_root = Path(__file__).resolve().parent

# images for face detection
custom_images = ["Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/istockphoto.jpg", "Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/people.jpg", "Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/Mona_Lisa.jpg"]
inference_dir = project_root / "inference_results"
# Setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Modell initialisieren (gleich wie vorher)
model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
num_classes = 2  # Hintergrund + Gesicht
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# → NEUESES: Fine‑getuntes Modell laden
checkpoint_path = project_root / "checkpoints" / "fasterrcnn_mobilenet_v3_finetuned.pth"
if checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Geladene Gewichte von {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint nicht gefunden unter {checkpoint_path}")

run_inference(model, custom_images, inference_dir, device)