

import os
import cv2
import numpy as np
from datetime import datetime
import face_recognition
from flask import Flask, render_template, Response, request, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from PIL import Image
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO or ERROR for production
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Logs to console, you can add a file handler if needed
)

# Initialize Flask app
app = Flask(__name__)

# Secret key for session management
app.secret_key = os.urandom(24)

# Define folders for saving known faces and uploaded images
KNOWN_FACES_FOLDER = "known_faces"
UPLOAD_FOLDER = "uploads"
attendance_file = "attendance.csv"

# Ensure folders exist
os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Admin credentials (stored in environment variables)
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')  # Default to 'admin' if not set in .env
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH', generate_password_hash('password123'))

# Load known faces and names
def load_known_faces():
    known_faces = []
    known_names = []
    if not os.path.exists(KNOWN_FACES_FOLDER):
        os.makedirs(KNOWN_FACES_FOLDER)
    for filename in os.listdir(KNOWN_FACES_FOLDER):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(KNOWN_FACES_FOLDER, filename)
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_img)
            if face_encodings:
                known_faces.append(face_encodings[0])
                known_names.append(filename.split('.')[0])  # Name is the file name without extension
    return known_faces, known_names

known_faces, known_names = load_known_faces()

# Initialize Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Check if the provided credentials are correct
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['logged_in'] = True
            logging.info(f"Admin logged in: {username}")
            return redirect(url_for('register_face'))  # Redirect to register face page
        else:
            logging.warning(f"Failed login attempt: {username}")
            return render_template('login.html', error="Invalid credentials")
    
    return render_template('login.html')

@app.route('/login2', methods=["GET", "POST"])
def login2():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Check if the provided credentials are correct
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['logged_in'] = True
            logging.info(f"Admin logged in: {username}")
            return redirect(url_for('delete_face'))  # Redirect to register face page
        else:
            logging.warning(f"Failed login attempt: {username}")
            return render_template('login.html', error="Invalid credentials")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Remove the 'logged_in' session key
    logging.info("Admin logged out.")
    return redirect(url_for('login'))  # Redirect to login page after logout

@app.route('/mark-attendance', methods=["GET", "POST"])
def mark_attendance():
    # This route is now accessible without login
    if request.method == "POST":
        # Handle attendance marking
        name = request.form.get('name')
        action = request.form.get('action')

        # Ensure that both 'name' and 'action' are selected
        if not name or not action:
            return render_template("mark_attendance.html", known_names=known_names, error="Please select both name and action.")
        
        # Mark attendance
        mark_attendance_in_csv(name, action)
        logging.info(f"Attendance marked for {name} with action {action}")
        return render_template("mark_attendance.html", known_names=known_names, action=action, name=name)
    
    return render_template('mark_attendance.html', known_names=known_names)

@app.route('/register-face', methods=["GET", "POST"])
def register_face():
    # Ensure that the user is logged in before accessing the register face page
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login page if not logged in
    
    if request.method == "POST":
        # Handle face registration
        file = request.files['image']
        name = request.form.get('name')

        if file and name:
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Allow only image files
                # Check if the name already exists
                if name in known_names:
                    return render_template('register_face.html', message=f"Name '{name}' already registered. Please use a different name.", msg_type="danger")
                
                # Save image
                filename = secure_filename(file.filename)
                img_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(img_path)

                # Extract face encoding
                image = Image.open(img_path)
                img = np.array(image)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                face_encodings = face_recognition.face_encodings(rgb_img)

                if face_encodings:
                    known_faces.append(face_encodings[0])  # Add the new face encoding to the list
                    known_names.append(name)
                    file.save(os.path.join(KNOWN_FACES_FOLDER, f"{name}.jpg"))
                    logging.info(f"Face registered successfully for {name}")
                    return render_template('register_face.html', message=f"Face of '{name}' registered successfully!", msg_type="success")
                else:
                    logging.error(f"No face detected in the uploaded image for {name}.")
                    return render_template('register_face.html', message="No face detected in the uploaded image. Please try again.", msg_type="danger")
            else:
                return render_template('register_face.html', message="Invalid file format. Please upload a valid image file (.png, .jpg, .jpeg).", msg_type="warning")
        
        logging.warning("File or name not provided.")
        return render_template('register_face.html', message="Please provide both a valid image and a name.", msg_type="warning")
    
    return render_template('register_face.html')

@app.route('/view-attendance')
def view_attendance():
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        return render_template("view_attendance.html", tables=[df.to_html(classes='data')], titles=df.columns.values)
    else:
        return render_template("view_attendance.html", message="No attendance records found.")

@app.route('/download-attendance')
def download_attendance():
    if os.path.exists(attendance_file):
        return send_file(attendance_file, as_attachment=True)
    else:
        return redirect(url_for('view_attendance'))

@app.route('/delete-face', methods=["GET", "POST"])
def delete_face():
    # Ensure that the user is logged in before accessing the delete face page
    if 'logged_in' not in session:
        return redirect(url_for('login2'))  # Redirect to login page if not logged in

    if request.method == "POST":
        name_to_delete = request.form.get('name')

        if not name_to_delete:
            return render_template('delete_face.html', message="Please select a name to delete.", msg_type="danger", known_names=known_names)

        if name_to_delete in known_names:
            try:
                index_to_delete = known_names.index(name_to_delete)
                known_faces.pop(index_to_delete)
                known_names.pop(index_to_delete)

                face_image_path = os.path.join(KNOWN_FACES_FOLDER, f"{name_to_delete}.jpg")
                if os.path.exists(face_image_path):
                    os.remove(face_image_path)
                    logging.info(f"Face image for '{name_to_delete}' deleted successfully.")
                else:
                    logging.warning(f"Image file for '{name_to_delete}' not found at path: {face_image_path}")

                logging.info(f"Face for '{name_to_delete}' deleted successfully.")
                return render_template('delete_face.html', message=f"Face of '{name_to_delete}' deleted successfully!", msg_type="success", known_names=known_names)

            except Exception as e:
                logging.error(f"Error deleting face for '{name_to_delete}': {str(e)}")
                return render_template('delete_face.html', message="An error occurred while deleting the face. Please try again.", msg_type="danger", known_names=known_names)

        else:
            logging.warning(f"Attempted to delete non-existing face: '{name_to_delete}'")
            return render_template('delete_face.html', message=f"Face '{name_to_delete}' not found.", msg_type="danger", known_names=known_names)

    return render_template('delete_face.html', known_names=known_names)


def mark_attendance_in_csv(name, action="entry"):
    # Check if the attendance file exists, if not create it with header
    if not os.path.exists(attendance_file):
        with open(attendance_file, "w", newline="") as f:
            f.write("Serial Number,Name,Time,Action\n")  # Write header with action column for punch-in/out

    # Prevent duplicate attendance for the same day
    try:
        df = pd.read_csv(attendance_file)
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Ensure the file has the correct columns
        if 'Name' not in df.columns or 'Time' not in df.columns:
            raise KeyError("Missing required columns in the attendance file.")
        
        if any((df['Name'] == name) & (df['Time'].str.contains(today))):  # Check for duplicate attendance
            logging.info(f"Attendance already marked for {name} today.")
            return  # Skip marking attendance if already marked for the day
    except (pd.errors.EmptyDataError, KeyError):
        # Handle the case where the file is empty or doesn't have the required columns
        df = pd.DataFrame(columns=["Serial Number", "Name", "Time", "Action"])
        today = datetime.now().strftime('%Y-%m-%d')
    
    # Generate serial number for the new entry
    serial_number = len(df) + 1

    # Write attendance entry to the CSV
    with open(attendance_file, "a", newline="") as f:
        f.write(f"{serial_number},{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{action}\n")
        logging.info(f"Attendance marked for {name} with action {action}")

# Video stream generation
def generate_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize the frame for faster processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Process the frame and recognize faces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)

            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            # Draw rectangles around faces and label them
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)






