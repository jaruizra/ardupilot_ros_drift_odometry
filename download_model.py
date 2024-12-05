from huggingface_hub import hf_hub_download

# Download the YOLOv8m-building-segmentation model
model_path = hf_hub_download(repo_id="keremberke/yolov8s-building-segmentation", filename="best.pt")

print(f"Model downloaded to: {model_path}")
