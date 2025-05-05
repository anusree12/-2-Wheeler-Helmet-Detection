from ultralytics import YOLO
import cv2
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from paddleocr import PaddleOCR

# Load YOLO model and PaddleOCR once
trained_model = YOLO("runs/detect/helmet_detection/weights/best.pt")  # docs: train_model.py


ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.3, rec_algorithm='CRNN')
class_names = ["number plate", "rider", "with helmet", "without helmet"]

#.....................................................................

# Helper functions
# Gets the center point of a bounding box.
def get_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2

#.....................................................................

#Checks if the inner box is fully inside the outer box — used to check
# if a number plate is inside a rider box or helmetless box inside a rider.
def is_inside(inner, outer):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

#....................................................................

#Fixes OCR mistakes like 'O' → '0', 'S' → '5' —
#helps normalize number plate output.

def correct_common_ocr_errors(text):
    replacements = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2'}
    return ''.join(replacements.get(c, c) for c in text)

#....................................................................

# Crops the number plate area with a 15-pixel padding.
# Performs OCR on the cropped image.
# Cleans and corrects OCR text.
# Returns both raw OCR output and corrected plate.

def number_plate_det(res, plate_box, plate_index=0):
    output = []
    x1, y1, x2, y2 = map(int, plate_box.xyxy[0])
    padding = 15
    x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
    x2 = min(x2 + padding, res.orig_img.shape[1])
    y2 = min(y2 + padding, res.orig_img.shape[0])
    crop = res.orig_img[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        output.append(f"⚠️ Plate {plate_index + 1} - Cropped image too small or empty, skipping.")
        return '\n'.join(output)

    ocr_result = ocr.ocr(crop, cls=True)
    if ocr_result and isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
        full_text = ''.join(line[1][0].strip().replace(" ", "") for line in ocr_result[0])
        corrected_text = correct_common_ocr_errors(full_text.upper())
        output.append(f"🚨 Plate {plate_index + 1} - OCR Raw: {full_text}")
        output.append(f"🧹 Plate {plate_index + 1} - Corrected: {corrected_text}")
    else:
        output.append(f"❌ Plate {plate_index + 1} - OCR failed or returned no usable text.")

    return '\n'.join(output)

#......................................................................



import cv2
import os


def run_detection(image_path):
    output_log = []

    # Inference
    result = trained_model(image_path)[0]
    helmetless_boxes = []
    number_plate_boxes = []

    #Separate Detected Boxes
    for box in result.boxes:
        class_id = int(box.cls)
        label = class_names[class_id]
        if label == "without helmet":
            helmetless_boxes.append(box)
        elif label == "number plate":
            number_plate_boxes.append(box)

    if not helmetless_boxes:
        return "✅ All riders appear to be wearing helmets.", None


    # looping through each detected "helmetless rider" (helmetless_box) and trying to find the
    # corresponding full "rider" box that contains the helmetless area.
    for i, helmetless_box in enumerate(helmetless_boxes):
        matched_rider_box = None

        for rider_box in result.boxes:
            #Match Helmetless Box to Rider Box
            if int(rider_box.cls) == class_names.index("rider"):
                #Check if the helmetless box is fully inside the rider box using:
                if is_inside(helmetless_box.xyxy[0], rider_box.xyxy[0]):
                    matched_rider_box = rider_box
                    break

        if not matched_rider_box:
            output_log.append(f"\n⚠️ Helmetless Rider {i + 1} - No enclosing rider box found.")
            continue

        rider_xyxy = matched_rider_box.xyxy[0]  # extract the coordinates of the matched rider box for the current helmetless rider.

        # Find Number Plates Inside That Rider (Loops through all detected number plates.
        # Checks: "Is this plate completely inside the rider's bounding box?"
        # If yes → Add it to inside_plates
        inside_plates = [plate_box for plate_box in number_plate_boxes if is_inside(plate_box.xyxy[0], rider_xyxy)]

        #If Plates Are Found Inside
        if inside_plates:
            rider_cx, rider_cy = get_center(helmetless_box.xyxy[0]) #find the nearest plate to the helmetless person.

            #Choose Closest Plate Based on Distance
            #For each plate inside the rider box, calculate the distance to the helmetless rider’s center.
            #Select the closest plate using min() — this is likely the actual number plate of that helmetless rider.
            def distance_to_rider_center(plate_box):
                px, py = get_center(plate_box.xyxy[0])
                return ((rider_cx - px) ** 2 + (rider_cy - py) ** 2) ** 0.5


            matched_plate = min(inside_plates, key=distance_to_rider_center)

            output_log.append(f"\n📸 Helmetless Rider {i + 1} - Number plate detected inside rider bounding box")
            output_log.append(number_plate_det(result, matched_plate, i))
        else:
            output_log.append(f"\n❌ Helmetless Rider {i + 1} - No number plate detected inside rider bounding box")

    # Save the annotated result image
    output_image_path = os.path.join("static", "uploads", "output.jpg")
    result.save(output_image_path)  # This saves the image with boxes

    return '\n'.join(output_log), output_image_path.replace("\\", "/")
