import threading
import time
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import pymongo
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import joblib
import base64
import io
import db_helper

load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize face recognition models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0).to(device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load trained model
try:
    model = joblib.load("face_model.pkl")
    print("✅ Face recognition model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# MongoDB Setup
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(mongo_uri)
db = client["attendance_system"]
collection = db["logs"]

# Initialize database and cloud storage
db_helper.init_db()

# Schedule Configuration
# Format: "HH:MM" (24-hour)
SCHEDULE = [
    {"subject": "AIML",         "start": "10:21", "end": "10:24", "grace_period": 2},
    {"subject": "Data Science", "start": "10:25", "end": "10:28", "grace_period": 2},
]

# Track attendance per subject
subject_attendance = {cls["subject"]: set() for cls in SCHEDULE}
processed_absentees = {cls["subject"]: False for cls in SCHEDULE}

# Load expected students
DATASET_DIR = "dataset/"
if os.path.exists(DATASET_DIR):
    expected_students = set(os.listdir(DATASET_DIR))
else:
    print("Warning: dataset/ directory not found. Cannot determine expected students.")
    expected_students = set()

def get_current_class():
    now = datetime.now()
    current_time_str = now.strftime("%H:%M")
    
    for cls in SCHEDULE:
        # Simple string comparison works for "HH:MM" in 24h format
        if cls["start"] <= current_time_str < cls["end"]:
            # Calculate minutes elapsed since start
            start_dt = datetime.strptime(cls["start"], "%H:%M").replace(
                year=now.year, month=now.month, day=now.day)
            elapsed_minutes = (now - start_dt).total_seconds() / 60
            
            is_late = elapsed_minutes > cls["grace_period"]
            return cls, is_late, elapsed_minutes
            
    return None, False, 0

def check_and_mark_absentees():
    """Background task to check for class endings and mark absentees"""
    global processed_absentees
    
    while True:
        try:
            now = datetime.now()
            current_time_str = now.strftime("%H:%M")
            
            for cls in SCHEDULE:
                subject_name = cls["subject"]
                class_end_time = cls["end"]
                grace_period = cls["grace_period"]
                
                # Check if class just ended (within 1 minute after grace period)
                end_dt = datetime.strptime(class_end_time, "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day)
                grace_end_dt = end_dt + timedelta(minutes=grace_period)
                
                # If current time is past grace period and we haven't processed absentees yet
                if now > grace_end_dt and not processed_absentees[subject_name]:
                    processed_absentees[subject_name] = True
                    
                    # Mark absentees
                    present_students = subject_attendance[subject_name]
                    absent_students = expected_students - present_students
                    
                    print(f"\n--- Class Ended: {subject_name} ---")
                    if absent_students:
                        print(f"Marking {len(absent_students)} students absent:")
                        for student in absent_students:
                            print(f"- {student} (Absent)")
                            db_helper.log_attendance(student, "Absent", subject_name)
                            
                            # Emit real-time update for absent students
                            socketio.emit('new_log', {
                                "name": student,
                                "status": "Absent",
                                "subject": subject_name,
                                "timestamp": datetime.now().isoformat(),
                                "confidence": 0
                            })
                    else:
                        print("All students present!")
                    print(f"--- End of {subject_name} Processing ---\n")
                    
        except Exception as e:
            print(f"Error in absentee checking: {e}")
        
        time.sleep(30)  # Check every 30 seconds

def get_embedding_from_frame(frame):
    """Extract face embedding from frame"""
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)
        if face is None:
            return None
        
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
        
        return embedding.flatten()
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

def recognize_face(frame):
    """Recognize face from frame"""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    
    # Get current class status
    current_class, is_late, elapsed_mins = get_current_class()
    
    embedding = get_embedding_from_frame(frame)
    if embedding is None:
        return {
            "status": "no_face", 
            "message": "No face detected",
            "class_info": get_class_status_info(current_class, is_late, elapsed_mins)
        }
    
    try:
        prediction = model.predict([embedding])[0]
        confidence = max(model.predict_proba([embedding])[0])
        
        if confidence > 0.7:  # Confidence threshold
            # Handle attendance marking based on class schedule
            attendance_status = "Present"
            message = f"{prediction} (Present)"
            can_mark_attendance = False
            
            if current_class and not is_late:
                subject_name = current_class["subject"]
                if prediction not in subject_attendance[subject_name]:
                    subject_attendance[subject_name].add(prediction)
                    print(f"Present: {prediction} marked for {subject_name}.")
                    attendance_status = "Present"
                    message = f"{prediction} marked for {subject_name}"
                    can_mark_attendance = True
                else:
                    message = f"{prediction} already marked for {subject_name}"
                    attendance_status = "Already Marked"
            elif current_class and is_late:
                attendance_status = "Late"
                message = f"{prediction} - Late! Not Marked"
            elif not current_class:
                attendance_status = "No Class"
                message = f"{prediction} - No active class"
            
            return {
                "status": "success",
                "name": prediction,
                "confidence": float(confidence),
                "message": message,
                "attendance_status": attendance_status,
                "can_mark_attendance": can_mark_attendance,
                "class_info": get_class_status_info(current_class, is_late, elapsed_mins)
            }
        else:
            return {
                "status": "unknown",
                "message": "Face not recognized",
                "confidence": float(confidence),
                "class_info": get_class_status_info(current_class, is_late, elapsed_mins)
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_class_status_info(current_class, is_late, elapsed_mins):
    """Get formatted class status information"""
    if current_class:
        if is_late:
            return {
                "subject": current_class["subject"],
                "status": "LATE - Closed",
                "color": "red"
            }
        else:
            time_left = int((current_class["grace_period"] * 60) - (elapsed_mins * 60))
            return {
                "subject": current_class["subject"],
                "status": f"Time Left: {time_left}s",
                "color": "green"
            }
    else:
        return {
            "subject": "No Active Class",
            "status": "Waiting for next class",
            "color": "yellow"
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mobile')
def mobile():
    return render_template('mobile.html')

@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        # Get image data from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"status": "error", "message": "No image data"}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"status": "error", "message": "Invalid image data"}), 400
        
        # Recognize face
        result = recognize_face(frame)
        
        # Log attendance only if face is recognized and can mark attendance
        if result["status"] == "success" and result.get("can_mark_attendance", False):
            name = result["name"]
            confidence = result["confidence"]
            class_info = result.get("class_info", {})
            subject = class_info.get("subject", "Mobile Recognition")
            
            # Log to database
            db_helper.log_attendance(name, "Present", subject)
            
            # Save proof image to cloud storage with subject folder
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                db_helper.upload_image_to_supabase(image_base64, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", subject)
            except Exception as e:
                print(f"Error uploading to cloud: {e}")
            
            # Emit real-time update
            socketio.emit('new_log', {
                "name": name,
                "status": "Present",
                "subject": subject,
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence
            })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in recognize endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/admin')
def admin_panel():
    return render_template('admin.html')

@app.route('/api/admin/verify', methods=['POST'])
def verify_admin():
    data = request.json
    password = data.get('password', '')
    
    if password == 'AISeAttendence@123':
        return jsonify({"status": "success", "message": "Access granted"})
    else:
        return jsonify({"status": "error", "message": "Invalid password"}), 401

@app.route('/api/admin/schedule', methods=['POST'])
def update_schedule():
    global SCHEDULE, subject_attendance, processed_absentees
    
    data = request.json
    password = data.get('password', '')
    
    if password != 'AISeAttendence@123':
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    try:
        new_schedule = data.get('schedule', [])
        SCHEDULE = new_schedule
        
        # Reset attendance tracking for new schedule
        subject_attendance = {cls["subject"]: set() for cls in SCHEDULE}
        processed_absentees = {cls["subject"]: False for cls in SCHEDULE}
        
        return jsonify({"status": "success", "message": "Schedule updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/schedule', methods=['GET', 'POST'])
def handle_schedule():
    """Get or update the class schedule configuration"""
    global SCHEDULE, subject_attendance, processed_absentees
    
    try:
        if request.method == 'GET':
            # Return current schedule
            return jsonify({
                "status": "success",
                "schedule": SCHEDULE
            })
        
        elif request.method == 'POST':
            # Update schedule (Admin only)
            data = request.json
            if not data or data.get('password') != 'AISeAttendence@123':
                return jsonify({"status": "error", "message": "Invalid password"}), 401
            
            # Update schedule
            new_schedule = data.get('schedule', [])
            
            # Validate schedule format
            for cls in new_schedule:
                if not all(key in cls for key in ['subject', 'start', 'end', 'grace_period']):
                    return jsonify({"status": "error", "message": "Invalid schedule format"}), 400
            
            SCHEDULE = new_schedule
            
            # Reset tracking for new schedule
            subject_attendance = {cls["subject"]: set() for cls in SCHEDULE}
            processed_absentees = {cls["subject"]: False for cls in SCHEDULE}
            
            print("✅ Schedule updated successfully")
            return jsonify({"status": "success", "message": "Schedule updated"})
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/class-status')
def get_class_status():
    """Get current class status without needing image data"""
    try:
        current_class, is_late, elapsed_mins = get_current_class()
        class_info = get_class_status_info(current_class, is_late, elapsed_mins)
        
        return jsonify({
            "status": "success",
            "class_info": class_info,
            "current_time": datetime.now().strftime("%H:%M")
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "current_time": datetime.now().strftime("%H:%M")
        }), 500

@app.route('/api/logs')
def get_logs():
    # Fetch last 50 logs, sorted by newest first
    logs = list(collection.find({}, {'_id': 0}).sort("timestamp", -1).limit(50))
    return jsonify(logs)

@app.route('/webhook/update', methods=['POST'])
def webhook_update():
    data = request.json
    # Emit the new log to all connected clients
    socketio.emit('new_log', data)
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    # Start background thread for checking absentees
    absentee_thread = threading.Thread(target=check_and_mark_absentees, daemon=True)
    absentee_thread.start()
    
    socketio.run(app, debug=True, port=5000)
