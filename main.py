import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os
from playsound import playsound
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables
load_dotenv()
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Load models
yolo_model = YOLO("yolov8n.pt")
cnn_model = load_model("CNN_mobilenetv2_model.h5")

# COCO labels
with open("coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

# Classify cropped person image
def classify_frame(image):
    image = cv2.resize(image, (224, 224)) / 255.0
    image = np.expand_dims(image, axis=0)
    pred = cnn_model.predict(image)[0][0]
    label = "Violent" if pred > 0.4 else "Non-Violent"
    return label, pred

# Email sender
def send_email(image_path):
    msg = EmailMessage()
    msg["Subject"] = "⚠️ Violence Detected!"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content("Violent activity detected. Screenshot attached.")
    with open(image_path, "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="jpeg", filename="alert.jpg")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)

# Initialize variables
cap = cv2.VideoCapture(0)
alert_triggered = False
alarm_cooldown = 10  # seconds
last_alarm_time = 0
consecutive_violent_frames = 0
required_consecutive_frames = 3
DEBUG = True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = yolo_model(frame)[0]
    current_frame_has_violence = False
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if class_names[cls_id] != "person":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1 < 50) or (y2 - y1 < 50):
            continue
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue
        label, confidence = classify_frame(person_crop)
        if DEBUG:
            print(f"[DEBUG] Detected: {label} | Confidence: {confidence:.2f}")

        # Draw label
        color = (0, 0, 255) if label == "Violent" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if label == "Violent" and confidence > 0.4:
            current_frame_has_violence = True

    # Update frame streak counter
    if current_frame_has_violence:
        consecutive_violent_frames += 1
    else:
        consecutive_violent_frames = 0

    # Trigger only after required number of violent frames
    if consecutive_violent_frames >= required_consecutive_frames:
        current_time = time.time()
        if (not alert_triggered) or ((current_time - last_alarm_time) > alarm_cooldown):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"alert_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            print("[ALERT] 🚨 Confirmed violent activity across frames.")
            try:
                playsound("alarm.wav")
            except Exception as e:
                print(f"[ERROR] Alarm sound failed: {e}")
            send_email(image_path)
            alert_triggered = True
            last_alarm_time = current_time

        # Reset counter after alert
        consecutive_violent_frames = 0

    cv2.imshow("Violence Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
