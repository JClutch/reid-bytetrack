from ultralytics.models.yolo.model import YOLO
import ultralytics

# Load an official or custom model
model = YOLO('yolov8n-seg.pt') 

# Run inference
# results = model('test_videos/1.mp4', save=True)

# Perform tracking with the model
 # Tracking with default tracker
#results = model.track(source="1.mp4", show=True)  # Tracking with ByteTrack tracker
results = model.track(source="reid/test2.mp4", save=True, show=True, persist=True, tracker="bytetrack.yaml")