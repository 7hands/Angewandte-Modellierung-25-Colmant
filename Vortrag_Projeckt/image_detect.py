import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import scipy.io
import numpy as np

# === Dataset & Transforms ===
class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transforms=None):
        self.annots = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        rec = self.annots[idx]
        img = Image.open(rec['image_id']).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        boxes = torch.as_tensor(rec['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(rec['labels'], dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
        return img, target


def get_transforms(train=True):
    t = [T.ToTensor()]
    if train:
        t.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(t)


def load_wider_annotations(mat_path, images_root):
    """
    Load WIDER FACE annotations from .mat split file without invalid filtering.
    Filters out boxes that are out-of-bounds or too small.
    """
    data = scipy.io.loadmat(mat_path)
    events = [e[0] for e in data['event_list'][0]]
    file_lists = data['file_list'][0]
    bb_lists = data['face_bbx_list'][0]
    records = []
    for ei, event in enumerate(events):
        img_names = file_lists[ei].ravel()
        bbs = bb_lists[ei].ravel()
        for name_arr, bb in zip(img_names, bbs):
            name = str(name_arr[0]) if hasattr(name_arr, 'shape') else str(name_arr)
            if not name.lower().endswith('.jpg'):
                name += '.jpg'
            img_path = (images_root / event / name).resolve()
            # load image size for clamping
            try:
                with Image.open(img_path) as tmp:
                    width, height = tmp.size
            except:
                continue
            boxes = []
            for x, y, w, h in bb:
                x1, y1 = float(x), float(y)
                x2, y2 = x1 + float(w), y1 + float(h)
                # clamp to image
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                # filter too small or invalid
                if x2 > x1 + 5 and y2 > y1 + 5:
                    boxes.append([x1, y1, x2, y2])
            if boxes:
                records.append({'image_id': str(img_path), 'boxes': boxes, 'labels': [1]*len(boxes)})
    return records

def collate_fn(batch):
    return tuple(zip(*batch))

# === Model Utils ===
def get_model(num_classes, pretrained_backbone=True):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, pretrained_backbone=pretrained_backbone)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model


def train_model(model, data_loader, optimizer, scheduler, device, num_epochs=20):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")
    return model


def run_inference(model, image_paths, output_dir, device, threshold=0.3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    to_tensor = ToTensor()
    to_pil = ToPILImage()
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        t = to_tensor(img).to(device)
        with torch.no_grad():
            out = model([t])[0]
        boxes = out['boxes'].cpu()
        scores = out['scores'].cpu()
        keep = scores >= threshold
        if not keep.any():
            print(f"No detections above threshold for {p}")
            continue
        boxes = boxes[keep].round().int()
        img_cpu = (t.cpu() * 255).to(torch.uint8)
        img_boxes = draw_bounding_boxes(img_cpu, boxes, labels=['face'] * len(boxes), colors=['green'] * len(boxes), width=2)
        to_pil(img_boxes).save(Path(output_dir) / Path(p).name)


if __name__ == '__main__':
    root = Path(__file__).parent
    # Paths to .mat split and images
    split_mat = root / 'data' / 'widerface' / 'wider_face_annotations' / 'wider_face_split' / 'wider_face_train.mat'
    images_root = root / 'data' / 'widerface' / 'WIDER_train' / 'WIDER_train' / 'images'

    # --- Training ---
    annots = load_wider_annotations(split_mat, images_root)
    ds = ObjectDetectionDataset(annots, transforms=get_transforms(train=True))
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2, pretrained_backbone=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    model = train_model(model, dl, optimizer, scheduler, device, num_epochs=10)

    # Save weights
    ckpt = root / 'checkpoints' / 'fasterrcnn_mobilenet_v3_finetuned.pth'
    ckpt.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"Saved weights to {ckpt}")

    # --- Inference on custom images ---
    custom_imgs = [str(root / 'own_images' / fname) for fname in ['istockphoto.jpg', 'people.jpg', 'Mona_Lisa.jpg']]
    run_inference(model, custom_imgs, root / 'inference_results', device, threshold=0.3)