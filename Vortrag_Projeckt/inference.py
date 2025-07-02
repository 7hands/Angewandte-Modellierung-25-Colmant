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