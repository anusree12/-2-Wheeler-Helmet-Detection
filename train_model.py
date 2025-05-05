from ultralytics import YOLO

# Load a YOLOv8 model architecture - nano for speed or small for balance
model = YOLO('yolov8n.yaml')  # use 'yolov8s.yaml' if you want a slightly better model

# Train the model on your custom dataset
results = model.train(
    data='data.yaml',   # your custom config
    epochs=50,
    imgsz=640,
    batch=16,
    name='helmet_detection'
)

# ----------------------------------------------------------
# Use the YOLOv8n (nano) architecture.
#
# Train it using your custom dataset defined in data.yaml.
#
# For 50 epochs, with image size 640, batch size 16.
#
# Save the experiment under the runs/detect/helmet_detection/ directory.


# # note that : after Run
# Creating helmet_detection, helmet_detection2, helmet_detection3
# Each time you run .train() with a different name argument, Ultralytics creates a new experiment folder inside runs/detect/.
#
# Example:name='helmet_detection'
# run agin with : name='helmet_detection2' , name='helmet_detection3' etc
# Or if you re-run with the same name, it automatically adds a suffix
#
# And this folder conatain weight folder , i think after run helmet_detection contain weight folder
# so i choose this folder for next process ( refert  detect_helmet.py)




