import sys
from ultralytics import YOLO

try:
    print("Loading mudasir13cs/floorcad-yolov8n-seg...")
    # ultralytics supports loading directly from huggingface repo if format is supported,
    # or we can use hf_hub_download to get the exact .pt weights file
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(repo_id="mudasir13cs/floorcad-yolov8n-seg", filename="best.pt")
    print(f"Downloaded weights to: {weights_path}")
    model = YOLO(weights_path)
    print("Model loaded successfully!")
    print(f"Number of classes: {len(model.names)}")
    print(f"Class names: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    # Check if there is another filename or repo
    import traceback
    traceback.print_exc()
