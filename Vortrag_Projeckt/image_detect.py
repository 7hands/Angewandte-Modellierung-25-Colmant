import os
from pathlib import Path
import torch
from PIL import Image
import scipy.io
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# for the CNN we need functions to get the data 
class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transforms=None):
        """
        annotations: list of dicts with keys:
          - 'image_id': full path to image file
          - 'boxes': list of [x1, y1, x2, y2]
          - 'labels': list of label ints
        transforms: torchvision.transforms.Compose for images only
        """
        self.annots = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        record = self.annots[idx]
        img = Image.open(record['image_id']).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        boxes = torch.as_tensor(record['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(record['labels'], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img, target


def get_transforms(train):
    transforms_list = [T.ToTensor()]
    if train:
        transforms_list.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms_list)


def collate_fn(batch):
    return tuple(zip(*batch))

# Load data
def load_widerface_annotation(txt_path, images_root):
    """
    Liest WIDER-Face TXT (ohne Count‑Zeilen) ein,
    filtert invalid=1 und fehlende Dateien,
    und gibt Records zurück.
    """
    images_root = Path(images_root)
    records = []
    current_img = None
    boxes = []
    labels = []
    attributes = []

    with open(txt_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Wird’s ein neuer Bild-Block?
            if line.lower().endswith('.jpg'):
                # Alten Block abschließen
                if current_img and boxes:
                    img_path = images_root.joinpath(*Path(current_img).parts).resolve()
                    if img_path.is_file():
                        records.append({
                            'image_id': str(img_path),
                            'boxes': boxes,
                            'labels': labels,
                            'attributes': attributes
                        })
                    else:
                        print(f"[WARN] Datei fehlt, skippe: {img_path}")
                # Reset für neues Bild
                current_img = line
                boxes = []
                labels = []
                attributes = []
                continue

            # Box‑Zeile: x y w h blur expr illum invalid occ pose
            parts = line.split()
            if len(parts) < 10 or current_img is None:
                # Format‑Fehler oder kein Bild vorher – überspringen
                continue

            x, y, w, h = map(float, parts[:4])
            blur, expr, illum, invalid, occ, pose = map(int, parts[4:10])

            # invalid=1 → gesamten Block verwerfen
            if invalid == 1:
                current_img = None
                boxes = []
                labels = []
                attributes = []
                continue

            if w > 0 and h > 0:
                x1, y1 = x, y
                x2, y2 = x + w, y1 + h
                boxes.append([x1, y1, x2, y2])
                labels.append(1)
                attributes.append({
                    'blur': blur,
                    'expression': expr,
                    'illumination': illum,
                    'occlusion': occ,
                    'pose': pose
                })

    # Letzten Block anhängen
    if current_img and boxes:
        img_path = images_root.joinpath(*Path(current_img).parts).resolve()
        if img_path.is_file():
            records.append({
                'image_id': str(img_path),
                'boxes': boxes,
                'labels': labels,
                'attributes': attributes
            })
        else:
            print(f"[WARN] Datei fehlt, skippe: {img_path}")

    return records


def run_inference(model, image_paths, output_dir, device, threshold=0.5):
    """
    Run inference on a list of image files and save visualized outputs.
    """
    from torchvision.utils import draw_bounding_boxes # for he boxes around the faces
    from torchvision.transforms import ToTensor # to load the trained model.
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

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    # Paths
    wider_txt = project_root / "data" / "widerface" / "wider_face_annotations" / "wider_face_split" / "wider_face_train_bbx_gt.txt"
    images_root = project_root / "data" / "widerface" / "WIDER_train"  / "WIDER_train" / "images"

    # Load dataset
    train_records = load_widerface_annotation(wider_txt, images_root)
    print(f"Loaded {len(train_records)} training images")
    # load the data via the ObjectdetectionDataset class
    train_ds = ObjectDetectionDataset(train_records, transforms=get_transforms(train=True))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn) 
    
    # Model setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # uses GPu when possibly
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None) # A RCNN from Mobilenet that I finetune
    num_classes = 2 # either a face ore background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Load pretrained weights if available
    checkpoints_dir = project_root / "checkpoints"
    checkpoint_path = checkpoints_dir / "fasterrcnn_mobilenet_v3_finetuned.pth"
    # if checkpoint_path.exists():
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #     print(f"Loaded checkpoint from {checkpoint_path}")

    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 5 # 20 training cycles
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0 # loss function less is better
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item() # adds the loss of training classification 
        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(train_loader):.4f}")

    # Save final model
    checkpoints_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path) # saves the modell to the checkpoint folder
    print(f"Model weights saved to {checkpoint_path}")

    print("Training complete")

    # images for face detection
    custom_images = ["Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/istockphoto.jpg", "Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/people.jpg", "Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/common-emotions.jpg"]
    inference_dir = project_root / "inference_results"
    run_inference(model, custom_images, inference_dir, device)
