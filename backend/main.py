import cv2
import face_recognition
import numpy as np
import os
import threading
import queue
import base64
from io import BytesIO
from PIL import Image
import json
import time
import pickle
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Global Variables & Constants ---
FACES_DIR = "faces"
ENCODINGS_FILE = "encodings.pickle"
attendance_queue = queue.Queue()
present_today = set()

# --- Helper Classes ---

class VideoCamera:
    """
    A singleton class that captures video frames in a separate thread.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VideoCamera, cls).__new__(cls)
                    cls._instance.video = None
                    cls._instance.stopped = True
                    cls._instance.frame = None
        return cls._instance

    def start_camera(self):
        if not self.stopped:
            return # Already running

        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Error: Could not open video source.")
            return

        self.grabbed, self.frame = self.video.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print("Camera started.")

    def update(self):
        while not self.stopped:
            if self.video and self.video.isOpened():
                grabbed, frame = self.video.read()
                if grabbed:
                    self.grabbed = grabbed
                    self.frame = frame
                else:
                    self.stop_camera()
            else:
                time.sleep(0.1)

    def stop_camera(self):
        self.stopped = True
        try:
            if hasattr(self, 'thread'):
                self.thread.join(timeout=1.0)
        except:
            pass
        
        if self.video and self.video.isOpened():
            self.video.release()
        
        self.frame = None 
        print("Camera stopped.")

    def get_frame(self):
        return self.frame

class FaceRecognitionSystem:
    """
    Manages known faces and performs recognition in a background thread.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(FaceRecognitionSystem, cls).__new__(cls)
                    cls._instance.known_face_encodings = []
                    cls._instance.known_face_roll_nos = []
                    cls._instance.student_roster = {}
                    cls._instance.load_faces()
                    
                    # State for the latest detection results
                    cls._instance.current_locations = []
                    cls._instance.current_names = []
                    cls._instance.stopped = False
                    
                    # Start background processing thread
                    cls._instance.thread = threading.Thread(target=cls._instance.process_faces, args=())
                    cls._instance.thread.daemon = True
                    cls._instance.thread.start()
        return cls._instance

    def load_faces(self):
        """Loads faces from disk or cache."""
        if not os.path.exists(FACES_DIR):
            os.makedirs(FACES_DIR)

        if os.path.exists(ENCODINGS_FILE):
            try:
                print("Loading encodings from cache...")
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_roll_nos = data["roll_nos"]
                self.student_roster = data["roster"]
                print(f"Loaded {len(self.known_face_encodings)} faces from cache.")
                return
            except Exception as e:
                print(f"Error loading cache: {e}. Rebuilding...")

        print("Loading known faces from disk...")
        for filename in os.listdir(FACES_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    roll_no, name = os.path.splitext(filename)[0].split('_', 1)
                    name = name.replace("_", " ")
                    image_path = os.path.join(FACES_DIR, filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_roll_nos.append(roll_no)
                        self.student_roster[roll_no] = name
                        print(f"Loaded: {roll_no} - {name}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        self.save_cache()

    def save_cache(self):
        data = {
            "encodings": self.known_face_encodings,
            "roll_nos": self.known_face_roll_nos,
            "roster": self.student_roster
        }
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)

    def add_student(self, roll_no, name, image_array, filename):
        encodings = face_recognition.face_encodings(image_array)
        if not encodings:
            return False, "No face detected. Please try again."
        
        encoding = encodings[0]
        self.known_face_encodings.append(encoding)
        self.known_face_roll_nos.append(roll_no)
        self.student_roster[roll_no] = name
        self.save_cache()
        return True, "Success"

    def process_faces(self):
        """Background loop to process frames for face recognition."""
        camera = VideoCamera()
        while not self.stopped:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                if self.known_face_encodings:
                    # FIX: Tolerance lowered to 0.45 for stricter matching (less false positives)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.45)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            roll_no = self.known_face_roll_nos[best_match_index]
                            name = self.student_roster.get(roll_no, "Error")
                            
                            # Mark attendance
                            if roll_no not in present_today:
                                attendance_queue.put(roll_no)
                
                face_names.append(name)

            self.current_locations = face_locations
            self.current_names = face_names
            
            time.sleep(0.05) 

    def get_latest_results(self):
        return self.current_locations, self.current_names

# --- Background Workers ---

def attendance_worker():
    fr_system = FaceRecognitionSystem()
    while True:
        roll_no = attendance_queue.get()
        if roll_no not in present_today:
            present_today.add(roll_no)
            name = fr_system.student_roster.get(roll_no, 'Unknown')
            print(f"Attendance Marked: {roll_no} - {name}")
        attendance_queue.task_done()

# --- Flask Routes ---

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if username == 'admin' and password == 'password':
        return jsonify({"success": True})
    else:
        return jsonify({"success": False}), 401

@app.route('/students', methods=['GET'])
def get_students():
    fr_system = FaceRecognitionSystem()
    roster_list = [{"rollNo": rn, "name": n} for rn, n in fr_system.student_roster.items()]
    return jsonify(sorted(roster_list, key=lambda x: x['rollNo']))

@app.route('/register', methods=['POST'])
def register_student():
    data = request.json
    roll_no = data['rollNo']
    name = data['name']
    image_data = base64.b64decode(data['imageData'].split(',')[1])
    
    try:
        image = Image.open(BytesIO(image_data))
        rgb_image_array = np.array(image)
        
        fr_system = FaceRecognitionSystem()
        
        filename = f"{roll_no}_{name.replace(' ', '_')}.jpg"
        if not os.path.exists(FACES_DIR):
            os.makedirs(FACES_DIR)
        image.save(os.path.join(FACES_DIR, filename))
        
        success, msg = fr_system.add_student(roll_no, name, rgb_image_array, filename)
        
        if success:
            return jsonify({"success": True, "message": f"Student {name} registered."})
        else:
            return jsonify({"success": False, "error": msg}), 400
            
    except Exception as e:
        print(f"REGISTRATION ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def generate_frames():
    camera = VideoCamera()
    fr_system = FaceRecognitionSystem()
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
            
        locations, names = fr_system.get_latest_results()
        
        # Scale up locations (since we scaled down by 0.20)
        scale = 5 
        for (top, right, bottom, left), name in zip(locations, names):
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-scan', methods=['POST'])
def start_scan():
    # FIX: Clear previous attendance data before starting new scan
    global present_today
    present_today.clear()
    
    # FIX: Clear the queue so old detections don't pop up
    with attendance_queue.mutex:
        attendance_queue.queue.clear()
        
    return jsonify({"message": "Scan session started. Attendance list cleared."})

@app.route('/get-results', methods=['GET'])
def get_results():
    fr_system = FaceRecognitionSystem()
    present_list = [{"rollNo": rn, "name": fr_system.student_roster.get(rn)} for rn in present_today]
    return jsonify(sorted(present_list, key=lambda x: x['rollNo']))

@app.route('/camera/stop', methods=['POST'])
def stop_cam():
    VideoCamera().stop_camera()
    return jsonify({"success": True, "message": "Camera released"})

@app.route('/camera/start', methods=['POST'])
def start_cam():
    VideoCamera().start_camera()
    return jsonify({"success": True, "message": "Camera started"})

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize systems
    FaceRecognitionSystem() 
    VideoCamera().start_camera() # Ensure camera starts
    
    worker_thread = threading.Thread(target=attendance_worker, daemon=True)
    worker_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)