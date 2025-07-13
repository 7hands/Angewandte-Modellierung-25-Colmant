import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_widerface_annotation(txt_path, images_root):
    images_root = Path(images_root)
    records = []
    current_img = None
    boxes, labels, attributes = [], [], []

    with open(txt_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.lower().endswith('.jpg'):
                if current_img and boxes:
                    img_path = images_root.joinpath(*Path(current_img).parts).resolve()
                    if img_path.is_file():
                        records.append({'image_id': str(img_path), 'boxes': boxes, 'labels': labels, 'attributes': attributes})
                current_img = line
                boxes, labels, attributes = [], [], []
                continue
            parts = line.split()
            if len(parts) < 10 or current_img is None:
                continue
            x, y, w, h = map(float, parts[:4])
            blur, expr, illum, invalid, occ, pose = map(int, parts[4:10])
            if invalid == 1:
                current_img = None
                boxes, labels, attributes = [], [], []
                continue
            if w > 0 and h > 0:
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                boxes.append([x1, y1, x2, y2])
                labels.append(1)
                attributes.append({'blur': blur, 'expression': expr, 'illumination': illum, 'occlusion': occ, 'pose': pose})
    if current_img and boxes:
        img_path = images_root.joinpath(*Path(current_img).parts).resolve()
        if img_path.is_file():
            records.append({'image_id': str(img_path), 'boxes': boxes, 'labels': labels, 'attributes': attributes})
    return records

class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transforms=None):
        self.annots = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        record = self.annots[idx]
        img = Image.open(record['image_id']).convert("RGB")
        if self.transforms:
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

def run_inference(model, image_paths, output_dir, device, threshold=0.5):
    from torchvision.utils import draw_bounding_boxes
    from torchvision.transforms import ToTensor
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    to_tensor = ToTensor()
    for img_path in image_paths:
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
        save_path = output_dir / Path(img_path).name
        T.ToPILImage()(img_boxes).save(save_path)
        print(f"Saved inference result to {save_path}")

if __name__ == "__main__":
    # Pfade definieren (an Verzeichnisstruktur anpassen)
    project_root = Path(__file__).resolve().parent
    # Im WIDER-Face-Paket liegen die Ground-Truths im Unterordner wider_face_split
    train_txt = project_root / "data/widerface/wider_face_annotations/wider_face_split/wider_face_train_bbx_gt.txt"
    val_txt   = project_root / "data/widerface/wider_face_annotations/wider_face_split/wider_face_val_bbx_gt.txt"
        # Pfade zu Bildern anpassen (zwei Ebenen WIDER_train/WIDER_train)
    train_images = project_root / "data/widerface/WIDER_train/WIDER_train/images"
    val_images   = project_root / "data/widerface/WIDER_val/WIDER_val/images"



    train_records = load_widerface_annotation(train_txt, train_images)
    val_records   = load_widerface_annotation(val_txt, val_images)

    train_ds = ObjectDetectionDataset(train_records, transforms=get_transforms(train=True))
    val_ds   = ObjectDetectionDataset(val_records, transforms=get_transforms(train=False))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True,
                                               num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=4, shuffle=False,
                                               num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Modell konfigurieren
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=2)
    model.to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Early Stopping Parameter
    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0
    num_epochs = 10

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targs = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targs)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad(); losses.backward(); optimizer.step()
            train_loss += losses.item()
        lr_scheduler.step()
        avg_train = train_loss / len(train_loader)

        # Validation (im Modus train() für Loss-Berechnung, aber ohne Gradienten)
        model.train()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [img.to(device) for img in imgs]
                targs = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targs)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Early Stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), project_root / "checkpoints/best_model.pth")
            print("Validation loss improved – Modell gespeichert.")
        else:
            epochs_no_improve += 1
            print(f"Keine Verbesserung seit {epochs_no_improve} Epochen.")
            if epochs_no_improve >= patience:
                print("Early stopping ausgelöst – Training beendet.")
                break

    # Letzter Checkpoint
    torch.save(model.state_dict(), project_root / "checkpoints/final_model.pth")
    print("Training und Validierung abgeschlossen.")

