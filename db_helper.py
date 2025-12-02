import os
import cv2
import pymongo
import requests
from supabase import create_client, Client
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

mongo_client = None
db = None
attendance_collection = None
supabase: Client = None

def init_db():
    global mongo_client, db, attendance_collection, supabase
    
    # MongoDB Setup
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri:
        try:
            mongo_client = pymongo.MongoClient(mongo_uri)
            db = mongo_client["attendance_system"]
            attendance_collection = db["logs"]
            print("✅ Connected to MongoDB")
        except Exception as e:
            print(f"❌ MongoDB Connection Error: {e}")
    else:
        print("⚠️ MONGO_URI not found in .env")

    # Supabase Setup
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if supabase_url and supabase_key:
        try:
            supabase = create_client(supabase_url, supabase_key)
            print("✅ Connected to Supabase")
        except Exception as e:
            print(f"❌ Supabase Connection Error: {e}")
    else:
        print("⚠️ SUPABASE_URL or SUPABASE_KEY not found in .env")

def log_attendance(name, status, subject="Unknown"):
    if attendance_collection is not None:
        record = {
            "name": name,
            "status": status,
            "subject": subject,
            "timestamp": datetime.now()
        }
        try:
            attendance_collection.insert_one(record)
            print(f"📝 Logged to MongoDB: {name} - {status} ({subject})")
            
            # Trigger Real-Time Update
            try:
                # Convert datetime to string for JSON serialization
                record_json = record.copy()
                record_json["timestamp"] = record["timestamp"].isoformat()
                record_json["_id"] = str(record["_id"])
                
                requests.post("http://127.0.0.1:5000/webhook/update", json=record_json, timeout=1)
            except Exception as req_err:
                # Don't crash if server is down
                pass
                
        except Exception as e:
            print(f"❌ Failed to log to MongoDB: {e}")

def upload_proof(name, frame, subject="General"):
    if supabase is None:
        return

    try:
        # Convert frame to bytes
        _, buffer = cv2.imencode(".jpg", frame)
        image_bytes = buffer.tobytes()
        
        filename = f"{name}_{int(datetime.now().timestamp())}.jpg"
        bucket_name = "attendance-proofs"
        
        # Create subject-wise folder path
        safe_subject = subject.replace(" ", "_").replace("/", "-")
        folder_path = f"{safe_subject}/{filename}"
        
        # Upload with subject folder structure
        supabase.storage.from_(bucket_name).upload(
            file=image_bytes,
            path=folder_path,
            file_options={"content-type": "image/jpeg"}
        )
        print(f"☁️ Uploaded proof to Supabase: {safe_subject}/{filename}")
    except Exception as e:
        print(f"❌ Failed to upload to Supabase: {e}")

def upload_image_to_supabase(image_base64, filename, subject="General"):
    """Upload base64 image to Supabase storage organized by subject folders"""
    if supabase is None:
        print("⚠️ Supabase not initialized")
        return None

    try:
        import base64
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)
        
        bucket_name = "attendance-proofs"
        
        # Create subject-wise folder path
        # Clean subject name for folder (remove special characters)
        safe_subject = subject.replace(" ", "_").replace("/", "-")
        folder_path = f"{safe_subject}/{filename}"
        
        # Upload to Supabase with subject folder structure
        supabase.storage.from_(bucket_name).upload(
            file=image_bytes,
            path=folder_path,
            file_options={"content-type": "image/jpeg"}
        )
        print(f"☁️ Uploaded proof to Supabase: {safe_subject}/{filename}")
        
        # Get public URL
        try:
            public_url = supabase.storage.from_(bucket_name).get_public_url(folder_path)
            return public_url
        except Exception as url_err:
            print(f"⚠️ Could not get public URL: {url_err}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to upload to Supabase: {e}")
        return None
