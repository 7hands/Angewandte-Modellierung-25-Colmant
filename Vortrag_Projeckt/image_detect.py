import os
from pathlib import Path
import torch
from PIL import Image
import scipy.io
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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


def load_wider_annotations(mat_path, images_root):
    data = scipy.io.loadmat(mat_path)
    events = [e[0] for e in data['event_list'][0]]
    file_lists = data['file_list'][0]
    face_bbx_lists = data['face_bbx_list'][0]
    records = []
    for ei, event in enumerate(events):
        img_names = file_lists[ei].ravel()
        bb_lists = face_bbx_lists[ei].ravel()
        for img_name_arr, bb_info in zip(img_names, bb_lists):
            if isinstance(img_name_arr, (np.ndarray, list, tuple)):
                img_name = str(img_name_arr[0])
            else:
                img_name = str(img_name_arr)
            if not img_name.lower().endswith('.jpg'):
                img_name += '.jpg'
            img_path = (images_root / event / img_name).resolve()
            boxes = []
            for x, y, w, h in bb_info:
                if w <= 0 or h <= 0:
                    continue
                x1, y1 = float(x), float(y)
                x2, y2 = x1 + float(w), y1 + float(h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
            if not boxes:
                continue
            records.append({'image_id': str(img_path), 'boxes': boxes, 'labels': [1]*len(boxes)})
    return records


def run_inference(model, image_paths, output_dir, device, threshold=0.5):
    """
    Run inference on a list of image files and save visualized outputs.
    """
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
    project_root = Path(__file__).resolve().parent
    # Paths
    split_mat = project_root / "data" / "widerface" / "wider_face_annotations" / "wider_face_split" / "wider_face_train.mat"
    images_root = project_root / "data" / "widerface" / "WIDER_train"  / "WIDER_train" / "images"

    # Load dataset
    train_records = load_wider_annotations(split_mat, images_root)
    print(f"Loaded {len(train_records)} training images")
    train_ds = ObjectDetectionDataset(train_records, transforms=get_transforms(train=True))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Model setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Load pretrained weights if available
    checkpoints_dir = project_root / "checkpoints"
    checkpoint_path = checkpoints_dir / "fasterrcnn_mobilenet_v3_finetuned.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(train_loader):.4f}")

    # Save final model
    checkpoints_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model weights saved to {checkpoint_path}")

    print("Training complete")

    # Example inference
    custom_images = ["Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/istockphoto.jpg", "Y:/Angewandte Modellierung/Angewandte-Modellierung-25-Colmant/Vortrag_Projeckt/own_images/people.jpg"]
    inference_dir = project_root / "inference_results"
    run_inference(model, custom_images, inference_dir, device)
