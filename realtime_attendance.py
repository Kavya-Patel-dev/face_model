import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import joblib
import os
import time
import datetime
import db_helper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=160, margin=0).to(device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

model = joblib.load("face_model.pkl")  # SVM classifier



def get_embedding_from_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)
    if face is None:
        return None
    
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
    
    return embedding.flatten()

cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)

frame_count = 0
process_every_n_frames = 5  # Process 1 out of every 5 frames
last_text = "Initializing..."
last_color = (255, 255, 255)
recognized_people = set()

# Load expected students
DATASET_DIR = "dataset/"
if os.path.exists(DATASET_DIR):
    expected_students = set(os.listdir(DATASET_DIR))
else:
    print("Warning: dataset/ directory not found. Cannot determine expected students.")
    expected_students = set()

# Schedule Configuration
# Format: "HH:MM" (24-hour)
SCHEDULE = [
    {"subject": "AIML",         "start": "10:42", "end": "10:45", "grace_period": 2},
    {"subject": "Data Science", "start": "10:45", "end": "10:48", "grace_period": 2},
]


def get_current_class():
    now = datetime.datetime.now()
    current_time_str = now.strftime("%H:%M")
    
    for cls in SCHEDULE:
        # Simple string comparison works for "HH:MM" in 24h format
        if cls["start"] <= current_time_str < cls["end"]:
            # Calculate minutes elapsed since start
            start_dt = datetime.datetime.strptime(cls["start"], "%H:%M").replace(
                year=now.year, month=now.month, day=now.day)
            elapsed_minutes = (now - start_dt).total_seconds() / 60
            
            is_late = elapsed_minutes > cls["grace_period"]
            return cls, is_late, elapsed_minutes
            
    return None, False, 0

# Initialize Database & Cloud
db_helper.init_db()

# Track attendance per subject
# Format: {"AIML": {"kavya", "person2"}, ...}
subject_attendance = {cls["subject"]: set() for cls in SCHEDULE}
processed_absentees = {cls["subject"]: False for cls in SCHEDULE}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Check Schedule
    current_class, is_late, elapsed_mins = get_current_class()
    
    if current_class:
        subject_name = current_class["subject"]
        
        # Handle "Time's Up" / Late logic
        if is_late and not processed_absentees[subject_name]:
            processed_absentees[subject_name] = True
            
            # Identify absentees for THIS subject
            present_students = subject_attendance[subject_name]
            absent_students = expected_students - present_students
            
            print(f"\n--- Attendance Closed for {subject_name} ---")
            if absent_students:
                print("Marked Absent:")
                for student in absent_students:
                    print(f"- {student}")
                    db_helper.log_attendance(student, "Absent", subject_name)
            else:
                print("All students present!")
            print("--------------------------------------\n")

    # UI Text
    if current_class:
        if is_late:
            status_text = f"Class: {current_class['subject']} (LATE - Closed)"
            status_color = (0, 0, 255)
        else:
            time_left = int((current_class["grace_period"] * 60) - (elapsed_mins * 60))
            status_text = f"Class: {current_class['subject']} (Time Left: {time_left}s)"
            status_color = (0, 255, 0)
    else:
        status_text = "No Active Class"
        status_color = (255, 255, 0)

    cv2.putText(frame, status_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        # Use last known result for skipped frames
        cv2.putText(frame, last_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)
        cv2.imshow("Attendance AI (PyTorch)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Resize frame for faster processing (50% scale)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    emb = get_embedding_from_frame(small_frame)

    if emb is not None:
        pred = model.predict([emb])[0]
        prob = model.predict_proba([emb]).max()

        if prob > 0.70:  # Threshold for recognition
            last_text = f"{pred} (Present)"
            last_color = (0, 255, 0)
            
            if current_class and not is_late:
                subject_name = current_class["subject"]
                if pred not in subject_attendance[subject_name]:
                    print(f"Present: {pred} marked for {subject_name}.")
                    subject_attendance[subject_name].add(pred)
                    
                    # Save to DB and Cloud
                    db_helper.log_attendance(pred, "Present", subject_name)
                    db_helper.upload_proof(pred, frame, subject_name)
            elif current_class and is_late:
                 last_text = "Late! Not Marked."
                 last_color = (0, 0, 255)
        else:
            last_text = "Person Not Recognized"
            last_color = (0, 0, 255)
    else:
        last_text = "No Face Detected"
        last_color = (0, 255, 255)

    cv2.putText(frame, last_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, last_color, 2)
    
    cv2.imshow("Attendance AI (PyTorch)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
