from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('detect_pills_22dec.pt')

# Set the path to the single image
# results = model('test_pill.jpg', save=True, conf=0.5)
results = model(source=0, show=True, conf=0.5)

# View results
for r in results:
    print(r.boxes.xywhn.tolist()[0])

# cd C:\Users\fangg\anaconda3_new\envs\.venv\Scripts
# activate
# cd C:\Users\fangg\OneDrive\Documents\FYP\FSP_IVideoAnalytics\Andrew\Pill-Detection\Notebooks\FYP_yolov8
# python test.py