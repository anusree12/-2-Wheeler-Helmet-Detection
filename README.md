🏍️ 2-Wheeler Helmet Detection with YOLOv8 & OCR (Web App)
This project uses YOLOv8 (yolov8n.pt) to detect helmet violations in 2-wheeler riders from uploaded images or videos. If a rider is detected without a helmet, it uses OCR (PaddleOCR) to extract the number plate from the image.


🚀 Features
Detects:

    Rider, With Helmet, Without Helmet, Number Plate
    Identifies riders without helmets
    Extracts vehicle number from number plates using OCR
    Web interface for uploading and viewing results

🛠️ Technologies Used

        YOLOv8 (yolov8n.pt)
        PaddleOCR
        Flask
        HTML/CSS (for frontend)
        Python 

📁 Project Structure
            
            helmet_detection_annot/
            ├── app.py                    # Flask application
            ├── detect_helmet.py          # YOLO + OCR detection logic
            ├── yolov8n.pt                # YOLOv8 model
            ├── datasets/                 # (Optional) custom dataset
            ├── runs/                     # YOLO inference outputs
            ├── static/
            │   ├── result_image/         # Output images with detections
            │   └── uploads/              # Uploaded files
            ├── templates/
            │   └── index.html            # Web UI template
            

🧠 Model Used

      YOLOv8n.pt: Lightweight YOLOv8 model trained/fine-tuned to detect:
      
      With Helmet , Without Helmet , Rider ,Number Plate

🌐 How to Run the Web App

      Clone this repository
      
      Install dependencies:
      
            pip install -r requirements.txt
            
      Run the app:
      
            python app.py
      
      Open in browser: http://localhost:5000
