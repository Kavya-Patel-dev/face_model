# Mobile Face Recognition Attendance System

## Quick Setup & Testing

### 1. Install Dependencies
```bash
pip install flask flask-socketio opencv-python torch torchvision facenet-pytorch joblib pymongo python-dotenv supabase pillow numpy
```

### 2. Environment Setup
Create a `.env` file with your credentials:
```
MONGO_URI=mongodb://localhost:27017/
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 3. Start the Server
```bash
python app.py
```

### 4. Access the Application
- **Desktop Dashboard**: http://localhost:5000
- **Mobile Camera**: http://localhost:5000/mobile

### 5. Mobile Testing
1. Open http://localhost:5000/mobile on your phone browser
2. Allow camera permission
3. Position your face in the camera
4. The system will auto-scan every 3 seconds
5. Results appear in real-time on both mobile and desktop dashboard

### Key Features
✅ **Real-time Face Recognition** - Mobile browser camera access  
✅ **Cloud Storage** - Automatic proof upload to Supabase  
✅ **Database Logging** - MongoDB attendance records  
✅ **Live Updates** - WebSocket real-time dashboard  
✅ **ERP Integration Ready** - API endpoints for ShikshaSetu  

### API Endpoints
- `POST /api/recognize` - Face recognition from base64 image
- `GET /api/logs` - Fetch attendance logs  
- `GET /mobile` - Mobile camera interface
- `GET /` - Desktop dashboard

### How It Works
1. **Camera Capture** → Mobile browser accesses camera
2. **Face Detection** → MTCNN detects faces in frames  
3. **Recognition** → FaceNet + SVM classifier identifies person
4. **Cloud Upload** → Proof image saved to Supabase
5. **Database Log** → Attendance recorded in MongoDB
6. **ERP Update** → Real-time data available for ShikshaSetu integration

### Mobile Browser Compatibility
- ✅ Chrome (Android/iOS)
- ✅ Safari (iOS) 
- ✅ Firefox (Android)
- ✅ Edge (Android)

**Note**: Make sure your trained model `face_model.pkl` exists in the project directory.