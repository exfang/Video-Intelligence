from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session, Response, send_from_directory,send_file, g
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Time, text, and_
from datetime import datetime, time, timedelta, timezone
import time, shutil, math, cv2, os, smtplib, sqlite3, base64, pygame, Augmentor, yaml, glob
from werkzeug.utils import secure_filename
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
from sklearn.model_selection import train_test_split
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, validators
from werkzeug.security import generate_password_hash, check_password_hash
import supervision as sv # pip install supervision==0.8.0
from collections import Counter
import plotly.express as px
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

app = Flask(__name__)
PILL_FOLDER = './pill_folder'
UPLOAD_FOLDER = './uploaded'
MEMORY_FOLDER = './memory_faces'
SCREENSHOT_FOLDER = './screenshot_folder'
app.secret_key = 'notsecret'
app.config['SCREENSHOT_FOLDER'] = SCREENSHOT_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PILL_FOLDER'] = PILL_FOLDER
app.config['MEMORY_FOLDER'] = MEMORY_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test5.db'
db = SQLAlchemy(app)
pygame.mixer.init()

pill_detection_model = YOLO('./models/pill_detection/pill_detection_medium_8batch.pt')
model = YOLO('./models/fall_detection/best_80.pt')
classNames = ["Fall"]

try:
    cap = cv2.VideoCapture(0)
except:
    raise("No camera detected")

fall_detected = False
confidence_value = 0.0
display_text = ""

recordings_folder = os.path.join('recordings')
screenshot_folder = os.path.join('screenshot_folder')

# Move the context creation to the beginning of the script
app.app_context().push()

if not os.path.exists(PILL_FOLDER):
    os.makedirs(PILL_FOLDER)

if not os.path.exists(MEMORY_FOLDER):
    os.makedirs(MEMORY_FOLDER)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class Routine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    pill = db.Column(db.String(50), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    displayed = db.Column(db.Boolean, default=True)


class GameResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entry = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    tries = db.Column(db.Integer, nullable=False)
    photo_count = db.Column(db.Integer, nullable=False)

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)

base_dir = os.path.abspath(os.path.dirname(__file__))
default_user_photo_path = os.path.normpath(os.path.join(base_dir, 'static', 'images', 'user.jpg'))

class Caregiver(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    relationship = db.Column(db.String(50), nullable=False)
    number = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    default_photo_path = default_user_photo_path
    photo_path = db.Column(db.String(255), default=default_photo_path)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)


class BacklogEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now, nullable=False)  # Updated timestamp to use datetime.utcnow

class RegistrationForm(FlaskForm):
    username = StringField('Username', [validators.DataRequired()])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.Length(min=6, message='Password must be at least 6 characters long')
    ])
    confirm_password = PasswordField('Confirm Password', [
        validators.DataRequired(),
        validators.EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Register')

with app.app_context(): # Create the SQL DB
    db.create_all()

def allowed_file(filename): # Allowed image extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.before_request
def before_request():
    g.user = None
    if 'user_id' in session:
        g.user = session['user_id']

users = []

@app.route('/', methods=['GET', 'POST'])
def login():
    if not users:
        session.clear()
        session['acc_type'] = "Guest"

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        users.append(user)

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['acc_type'] = 'Loggedin'
            return render_template('./fall_detection/index.html')
        else:
            return 'Invalid username or password'

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        new_user = User(username=form.username.data, password_hash=generate_password_hash(form.password.data))
        db.session.add(new_user)
        db.session.commit()
        return f'User {form.username.data} registered successfully!'

    return render_template('register.html', form=form)


@app.route('/content')
def content():
    return render_template('content.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/index')
def index():
    return render_template('./fall_detection/index.html')



@app.route('/video_feed') # done
def video_feed():
    return Response(detect_objects(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_pills', methods=['GET', 'POST'])
def upload_pills():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('image')
        category = request.form['category']

        # If the user does not select a file or category, return an error
        if not files or not category:
            flash('Please select at least one file and enter a category')
            return redirect(request.url)
        
        for file in files:
            if file and allowed_file(file.filename):
                img = Image.open(file.stream)
                results = pill_detection_model(img, conf=0.6)
                for r in results:
                    if len(r.boxes.xywhn.tolist()) == 1:
                        continue
                    elif len(r.boxes.xywhn.tolist()) > 1:
                        flash("Multiple pills detected. Please upload an image of a single pill.")
                        return redirect(request.url)
                    else:    
                        flash("No pills detected. Please reupload a new image")
                        return redirect(request.url)

        # Ensure the 'pill_folder' folder exists
        category_folder = os.path.join(app.config['PILL_FOLDER'], category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        # Save each file to the category folder
        for file in files:
            if file and allowed_file(file.filename):
                file.seek(0)
                # Use a secure filename to prevent malicious behavior
                secure_filename = os.path.join(category_folder, file.filename)
                file.save(secure_filename)
        
        flash('Files uploaded successfully')
        return redirect(request.url)

    return render_template('pill_detection/upload_pills.html')

@app.route('/configure_pills', methods=['GET', 'POST'])
def configure_pills():

    app.config['PILL_FOLDER'] = PILL_FOLDER
    # Get a list of image folders within the PILL_FOLDER
    image_folders = [folder for folder in os.listdir(app.config['PILL_FOLDER']) if os.path.isdir(os.path.join(app.config['PILL_FOLDER'], folder))]

    # Create a dictionary to store image names for each folder
    image_data = {}
    for folder in image_folders:
        folder_path = os.path.join(app.config['PILL_FOLDER'], folder)
        image_data[folder] = [image_name for image_name in os.listdir(folder_path) if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.jfif'))]
    
    return render_template('pill_detection/configure_pills.html', image_data=image_data, name='Pills')



@app.route('/configure_routine', methods=['GET', 'POST'])
def configure_routine():
    current_date = datetime.now().date()

    # Get a list of image folders within the PILL_FOLDER
    image_folders = [folder for folder in os.listdir(app.config['PILL_FOLDER']) if os.path.isdir(os.path.join(app.config['PILL_FOLDER'], folder))]

    # Create a dictionary to store image names for each folder
    image_data = {}
    for folder in image_folders:
        folder_path = os.path.join(app.config['PILL_FOLDER'], folder)
        image_data[folder] = [image_name for image_name in os.listdir(folder_path) if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    routine = Routine.query.filter_by(displayed=True)\
                      .filter(Routine.start_date <= current_date)\
                      .filter(Routine.end_date >= current_date)\
                      .order_by(Routine.start_date)\
                      .all()

    if request.method == 'POST':
        start_date_list = request.form.getlist('start_date')
        end_date_list = request.form.getlist('end_date')
        start_time_list = request.form.getlist('start_time')
        end_time_list = request.form.getlist('end_time')
        pill_list = request.form.getlist('pill')
        quantity_list = request.form.getlist('quantity')

        for start_date_str, end_date_str, start_time_str, end_time_str, pill, quantity in zip(start_date_list, end_date_list, start_time_list, end_time_list, pill_list, quantity_list):
            # Convert string representations to date and time objects
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            start_time = datetime.strptime(start_time_str, "%H:%M").time()
            end_time = datetime.strptime(end_time_str, "%H:%M").time()

            new_entry = Routine(start_date=start_date, end_date=end_date, start_time=start_time, end_time=end_time, pill=pill, quantity=quantity)
            db.session.add(new_entry)


        try:
            # Attempt to commit changes
            db.session.commit()
        except Exception as e:
            # Print or log the exception to understand the error
            print(f"Error committing changes to the database: {e}")

        return redirect(url_for('configure_routine'))

    return render_template('pill_detection/configure_routine.html', image_data=image_data, routine=routine)

@app.route('/hide_routine', methods=['POST'])
def hide_routine():
    routine_id = request.form.get('routine_id')
    
    # Add logic to update the displayed attribute of the routine
    routine_to_hide = Routine.query.get(routine_id)
    if routine_to_hide:
        routine_to_hide.displayed = False
        db.session.commit()
        flash(f'Routine with ID {routine_id} has been hidden successfully', 'success')
    else:
        flash(f'Routine with ID {routine_id} not found', 'error')

    return redirect(url_for('configure_routine'))


@app.route('/routine_history')
def routine_history():
    routine_history = Routine.query.order_by(Routine.start_date).all()
    return render_template('pill_detection/routine_history.html', routine_history=routine_history)


@app.route('/delete_routine', methods=['POST'])
def delete_routine():
    routine_id = request.form.get('routine_id')
    
    # Add logic to delete routine with the given ID from the database
    routine_to_delete = Routine.query.get(routine_id)
    if routine_to_delete:
        db.session.delete(routine_to_delete)
        db.session.commit()
        flash(f'Routine with ID {routine_id} has been deleted successfully', 'success')
    else:
        flash(f'Routine with ID {routine_id} not found', 'error')

    return redirect(url_for('routine_history'))

from sqlalchemy import func

@app.route('/medication_dashboard')
def medication_dashboard(filtered_start=None, filtered_end=None, selected_pill=None):
    # Fetch unique start/end date/time combinations from the database
    current_date = datetime.now().date()
    date_now = datetime.now().date()

    print("Current date:", current_date)
    filtered_start = request.args.get('filtered_start')
    filtered_end = request.args.get('filtered_end')
    selected_pill = request.args.get('selected_pill')
    print(selected_pill)
    if filtered_start or filtered_end:
        filtered_start = datetime.utcfromtimestamp(int(filtered_start)).strftime('%Y-%m-%d')
        filtered_end = datetime.utcfromtimestamp(int(filtered_end)).strftime('%Y-%m-%d')
    else:
        filtered_start = None
        filtered_end = None

    print("filtered_start", filtered_start)
    print("filtered_end", filtered_end)
    
    unique_datetime_combinations = db.session.query(
            Routine.start_date, Routine.end_date, Routine.start_time, Routine.end_time, Routine.pill
        ).filter(
            and_(
                Routine.start_date <= current_date
            )
        ).group_by(Routine.start_date, Routine.end_date, Routine.start_time, Routine.end_time).all()

    earliest_start_date = db.session.query(func.min(Routine.start_date)).scalar()

    unique_pills = db.session.query(Routine.pill).distinct().all()
    unique_pills = [pill[0] for pill in unique_pills]
    
    if filtered_start:
        # Convert filtered_start and filtered_end from Unix timestamps to datetime objects
        if selected_pill == 'all' or selected_pill == None:
            unique_datetime_combinations = db.session.query(
                Routine.start_date, Routine.end_date, Routine.start_time, Routine.end_time, Routine.pill
            ).filter(
                and_(
                    Routine.start_date <= current_date,
                    Routine.start_date >= filtered_start,  # Filter start_date >= filtered_start
                    Routine.end_date <= filtered_end       # Filter end_date <= filtered_end
                )
            ).group_by(Routine.start_date, Routine.end_date, Routine.start_time, Routine.end_time).all()
        else:
            unique_datetime_combinations = db.session.query(
                Routine.start_date, Routine.end_date, Routine.start_time, Routine.end_time, Routine.pill
            ).filter(
                and_(
                    Routine.start_date <= current_date,
                    Routine.start_date >= filtered_start,  # Filter start_date >= filtered_start
                    Routine.end_date <= filtered_end,       # Filter end_date <= filtered_end
                    Routine.pill == selected_pill
                )
            ).group_by(Routine.start_date, Routine.end_date, Routine.start_time, Routine.end_time).all()
        
    individual_records = []
    # Iterate through each tuple in the original list
    for start_date, end_date, start_time, end_time, _ in unique_datetime_combinations:
        current_date = start_date
        while current_date <= end_date:
            if current_date <= date_now:
                individual_records.append((current_date, start_time, end_time))
                current_date += timedelta(days=1)
            else:
                break
    
    individual_records = list(set(individual_records)) # remove duplicates if any

    # print('unique_datetime_combinations:',unique_datetime_combinations)
    

    filename_pattern = "recording_%Y_%m_%d_%H_%M.mp4"
    # Initialize a list to store video filenames and corresponding datetime objects
    video_files = []
    
    # Iterate through the files in the folder
    for filename in os.listdir(app.config['RECORDING_FOLDER']):
        # Construct the full filepath
        filepath = os.path.join(recordings_folder, filename)

        # Check if the file is a video based on the filename pattern
        if filename.endswith(".mp4") and os.path.isfile(filepath):
            # Parse the datetime information from the filename
            video_datetime = datetime.strptime(filename, filename_pattern)
            
            # Append the filename and datetime object to the list
            video_files.append(video_datetime)

    video_tuples = [(video.date(), video.time()) for video in video_files]
    # print('video:',video_tuples)

    # List to store non-overlapping dates
    matching_routines = []

    # Iterate through individual_records
    for recorded_date, recorded_time in video_tuples:
        # Check if the recorded date matches any routine date
        for routine in individual_records:
            
            if routine[0] == recorded_date and routine[1] <= recorded_time <= routine[2]:
                matching_routines.append(routine)
    
    # print('matching_routines:',matching_routines)
    missing_routines = list(set(individual_records) - set(matching_routines))
    # print("Non-overlapping dates:", missing_routines)

    # Create a Plotly figure
    fig = make_subplots(rows=1, cols=1)

    # Extract dates and times for missing routines as strings
    missing_dates_str = [record[0].strftime('%Y-%m-%d') for record in missing_routines]
    missing_times_str = [record[1].strftime('%H:%M:%S') for record in missing_routines]

    # Create a scatter plot for missing medications (Red)
    fig.add_trace(go.Scatter(
        x=missing_dates_str,
        y=missing_times_str,
        mode='markers',
        marker=dict(color='red', size=10),
        name='Missed'
    ))

    # Extract dates and times for consumed routines as strings
    matching_dates_str = [record[0].strftime('%Y-%m-%d') for record in matching_routines]
    matching_times_str = [record[1].strftime('%H:%M:%S') for record in matching_routines]

    # Create a scatter plot for consumed medications (Green)
    fig.add_trace(go.Scatter(
        x=matching_dates_str,
        y=matching_times_str,
        mode='markers',
        marker=dict(color='green', size=10),
        name='Consumed'
    ))

    fig.update_layout(title='Medication Consumption',
                  xaxis_title='Date',
                  yaxis_title='Time',
                  showlegend=True,
                  autosize=True,
                  yaxis=dict(categoryorder='category ascending'))  # Set category order for y-axis

    # Convert Plotly chart to HTML
    chart_html = fig.to_html(full_html=False)

    # List to store missed medication details
    missed_medications_details = []

    # Fetch missed medication details from the database
    for missed_date, start_time, end_time in missing_routines:
        # print('missing_routines:',missed_date, start_time, end_time)
        # Query the database to retrieve pill and quantity for the missed date
        missed_medication_details = db.session.query(Routine.pill, Routine.quantity).filter(
            and_(
                Routine.start_date <= missed_date,
                missed_date <= Routine.end_date,
                Routine.start_time == start_time,
                Routine.end_time == end_time  # Assuming end_date is the same for one-day routines
            )
        ).first()

        # print(missed_medication_details)
        if missed_medication_details:
            missed_medications_details.append({
                'date': missed_date.strftime('%Y-%m-%d'),
                'pill': missed_medication_details[0],
                'quantity': missed_medication_details[1]
            })
    
    # print(missed_medications_details)
    donut_chart_fig = go.Figure(go.Pie(
        labels=[record['pill'] for record in missed_medications_details],
        values=[record['quantity'] for record in missed_medications_details],
        hole=0.3,
        marker=dict(colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']),
    ))

    donut_chart_fig.update_layout(title='Missed Medication Names', showlegend=True)

    # Convert donut chart to HTML
    donut_chart_html = donut_chart_fig.to_html(full_html=False)

    kpi_missed_medications = round((len(missing_routines) / len(individual_records)) * 100, 2) if len(individual_records) > 0 else 0


    return render_template('pill_detection/dashboard.html', chart_html=chart_html,
                           donut_chart_html=donut_chart_html, kpi_missed_medications=kpi_missed_medications,
                           no_of_missing_routines=len(missing_routines), total_records=len(individual_records),
                           earliest_start_date=earliest_start_date, current_date=current_date, unique_pills=unique_pills)


# Pill Detection


box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

def get_ongoing_routines():
    current_time = datetime.now().time()
    current_date = datetime.now().date()

    ongoing_routines = Routine.query.filter(Routine.start_time <= current_time, Routine.end_time >= current_time, Routine.start_date <= current_date, Routine.end_date >= current_date).all()
    return ongoing_routines

def activate_capture():
    cap = cv2.VideoCapture(0)
    return cap

detect_folder = './runs/detect/train'
results_path = os.path.join(detect_folder, 'results.png')
csv_path = os.path.join(detect_folder, 'results.csv')

pill_model = False
training = False
last_epoch_value = 0

def initialize_model():
    global pill_model, training, last_epoch_value
    if os.path.exists(results_path):
        # Copy the best model to the designated location
        best_model_source = os.path.join(detect_folder, 'weights', 'best.pt').replace('\\', '/')
        pill_model = YOLO(best_model_source)
    elif os.path.exists(csv_path) and not os.path.exists(results_path):
        training = True
        df = pd.read_csv(csv_path)

        # Check the last value in the 'epoch' column
        last_epoch_value = df['                  epoch'].iloc[-1]
    

initialize_model() # first intialization of the model

@app.route('/pill_detection')
def pill_detection():
    ongoing_routines = get_ongoing_routines()
    global cap
        
    if pill_model:
        present = 'True'
        ret, frame = cap.read()
        if ret == False:
            cap = activate_capture()
            ret, frame = cap.read()
            
        # Get real-time pill counts
        result = pill_model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        pill_counts = Counter(pill_model.names[class_id] for _, _, _, class_id, _ in detections)
        # Ensure no additional pills are consumed

        all_pills_match = all(
            pill_name in [routine.pill for routine in ongoing_routines] and pill_counts[pill_name] == routine.quantity for routine in ongoing_routines
            for pill_name in pill_counts.keys()
        )

        

        return render_template('pill_detection/detect_pills.html',
                            ongoing_routines=ongoing_routines,
                            pill_counts=pill_counts,
                            all_pills_match=all_pills_match,
                            present=present)
    
    
    elif training:
        initialize_model()
        return render_template('pill_detection/detect_pills.html',
        ongoing_routines=ongoing_routines,
        present='Training', epoch=last_epoch_value)
    
    initialize_model()
    return render_template('pill_detection/detect_pills.html',
        ongoing_routines=ongoing_routines,
        present='False')

@app.route('/capture_screenshot', methods=['POST'])
def capture_screenshot():
    _, frame = cap.read()

    screenshot_dir = './screenshot_folder'

    # Create the directory if it doesn't exist
    os.makedirs(screenshot_dir, exist_ok=True)

    # Define the filename for the screenshot (you can use a timestamp or any unique identifier)

    session['timestamp'] = datetime.now().strftime('%Y_%m_%d_%H_%M')
    timestamp = session['timestamp']

    screenshot_filename = os.path.join(screenshot_dir, f'screenshot_{timestamp}.jpg')
    
    # Save the frame as an image file
    cv2.imwrite(screenshot_filename, frame)
    return jsonify({'message': 'Screenshot captured successfully'})

@app.route('/screenshot/<filename>')
def serve_screenshot(filename):
    screenshot_path = os.path.join('screenshots', filename)
    print(f"Attempting to serve screenshot from path: {screenshot_path}")
    return send_from_directory('screenshots', filename)

@app.route('/pill_screenshot/<filename>')
def serve_pill_screenshot(filename):
    screenshot_path = os.path.join('screenshots', filename)
    print(f"Attempting to serve screenshot from path: {screenshot_path}")
    return send_from_directory('screenshot_folder', filename)

@app.route('/pill_detection_stream')
def pill_detection_stream():
    def generate_frames():
        while True:
            ret, frame = cap.read()
            result = pill_model(frame)[0]
            detections = sv.Detections.from_yolov8(result)
            
            labels = [
                f"{pill_model.names[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ in detections
            ]

            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            
            frame_bytes = buffer.tobytes()

            # Yield the frame in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(1)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/record_medication')
def record_medication():
    return render_template('pill_detection/record_medication.html')

RECORDING_FOLDER = './recordings'
app.config['RECORDING_FOLDER'] = RECORDING_FOLDER
out = None
recording = False

if not os.path.exists(RECORDING_FOLDER):
    os.makedirs(RECORDING_FOLDER)

def medication_frames():
    global cap, out, recording

    while True:
        _, frame = cap.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break

        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        if recording and out is not None:
            out.write(frame)

    # Release resources when the generator ends
    cap.release()
    if out is not None:
        out.release()

@app.route('/medication_video_feed_route')
def medication_video_feed_route():
    return Response(medication_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording')
def start_recording():
    global out, recording
    if not recording:
        timestamp = session['timestamp'] 
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # doesnt work
        # fourcc = cv2.VideoWriter_fourcc(*'X264') # works
        # fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(f'./recordings/recording_{timestamp}.mp4', -1, 20.0, (640, 480))
        recording = True
        return jsonify(success=True)
    else:
        return jsonify(success=False, message='Recording is already in progress')

@app.route('/stop_recording')
def stop_recording():
    global out, recording
    if recording:
        recording = False
        if out is not None:
            out.release()
            out = None
        return jsonify(success=True)
    else:
        return jsonify(success=False, message='No recording in progress')



# Function to format the date and time
def format_datetime(filename):
    # Split the filename using underscores
    parts = filename.split('_')
    
    # Extract date and time parts
    year = parts[1]
    month = parts[2]
    day = parts[3]
    hour = parts[4]
    minute = parts[5][:2]  # Take the first two characters for minutes

    # Format the datetime strings
    formatted_datetime = f"{day}/{month}/{year} {hour}:{minute}"

    return formatted_datetime

@app.route('/medication_recording')
def medication_recording():
    # Read the contents of the 'recordings' folder
    video_files = [file for file in os.listdir(recordings_folder) if file.endswith('.mp4')]

    # Format the date and time for each video
    video_list = [{'filename': filename, 'datetime': format_datetime(filename)} for filename in video_files]

    return render_template('pill_detection/medication_recording.html', video_list=video_list)

# Add a route to serve the static video files
@app.route('/recordings/<filename>')
def serve_video(filename):
    print(filename)
    return send_from_directory('recordings', filename)

@app.route('/delete_video_and_image/<filename>') # Route to delete a video
def delete_video_and_image(filename):
    try:
        
        video_path = os.path.join(recordings_folder, filename) # Delete the video
        os.remove(video_path)

        
        screenshot_filename = filename.replace('recording', 'screenshot').replace('.mp4', '.jpg') # Construct the corresponding image filename
        print(screenshot_filename)
        
        screenshot_path = os.path.join(screenshot_folder, screenshot_filename)
        os.remove(screenshot_path)

        return jsonify(success=True, message='Recording and associated image deleted successfully')
    except Exception as e:
        return jsonify(success=False, message=str(e))


# Retrain the model
@app.route('/retrain_model')
def retrain_model():
    try:
        global pill_model
        pill_model = False
        # Specify the paths to the folders and files you want to delete
        folders_to_delete = ['./Annotated', './Merged_Images', './runs']
        file_to_delete = './data.yaml'

        # Function to delete folders and their contents recursively
        def delete_folder_contents(folder_path):
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        os.rmdir(dir_path)
                os.rmdir(folder_path)

        # Delete specified folders and files
        for folder_path in folders_to_delete:
            if os.path.exists(folder_path):
                if folder_path == './Merged_Images':
                    delete_folder_contents(folder_path)
                else:
                    shutil.rmtree(folder_path)

        # Delete specified file
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)
        
        # Delete each output folder and its contents
        # Specify the path to the root folder
        root_folder = './pill_folder/'

        # Specify the wildcard pattern for the pill folders
        pill_folders_pattern = os.path.join(root_folder, '*')

        # Use glob to find all pill folders
        pill_folders = glob.glob(pill_folders_pattern)

        # Iterate over each pill folder and remove the 'output' folders
        for pill_folder in pill_folders:
            output_folder_path = os.path.normpath(os.path.join(pill_folder, 'output'))
            # print("output::",output_folder_path)
            # Check if the 'output' folder exists and remove it
            if os.path.exists(output_folder_path):
                shutil.rmtree(output_folder_path)
        
        # Delete the file
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)

        # Optionally, you can redirect to another route after the deletion
        # Function to apply augmentation to each subfolder
        def augment_images_in_folder(folder_path, num_augmented_images):
            pipeline = Augmentor.Pipeline(folder_path)
            
            # Define augmentation operations
            pipeline.flip_left_right(probability=0.7)
            pipeline.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.4)
            pipeline.rotate90(probability=0.6)
            pipeline.zoom_random(probability=0.5, percentage_area=0.8)
            
            pipeline.sample(num_augmented_images)

        # Main script
        input_folder = './pill_folder/'

        # Iterate through subfolders and apply augmentation
        for subfolder in os.listdir(input_folder):
            subfolder_path = os.path.join(input_folder, subfolder)
            
            # Check if it's a directory
            if os.path.isdir(subfolder_path):
                print(f"Augmenting images in folder: {subfolder}")
                augment_images_in_folder(subfolder_path, 100)
                
                
        input_root_folder = './pill_folder/'
        output_root_folder = './Annotated/'
        augmented_folder = 'output'  # Adjust this according to the new structure
        data_yaml_path = 'data.yaml'

        # load the model
        yolo = YOLO('models/pill_detection/pill_detection_medium_8batch.pt')

        # update the data.yaml file with new classes
        def update_data_yaml(data_yaml_path, new_class):

            if not os.path.exists(data_yaml_path):  # Check if the file exists
                with open(data_yaml_path, 'w') as file:  # Create an empty YAML file with the necessary structure
                    yaml.dump({'nc': 0, 'names': [], 'train':'', 'test':'', 'val':''}, file)

            with open(data_yaml_path, 'r') as file:  # Load the existing YAML content
                data = yaml.load(file, Loader=yaml.FullLoader)

            if not data['names']:  # Check if the new class already exists (case-insensitive check)
                data['names'] = [new_class]  # Initialize the data['names'] to be a list
                data['nc'] = 1

                if not data['train'] and not data['test'] and not data['val']:
                    data['train'] = 'Merged_Images/train/images'
                    data['test'] = 'Merged_Images/test/images'
                    data['val'] = 'Merged_Images/val/images'
                    
                with open(data_yaml_path, 'w') as file:
                    yaml.dump(data, file)
                print(f"Class '{new_class}' added to {data_yaml_path}")

            
                
            else:  # Append new classes if data['names'] is already a list
                if new_class not in [existing_class.lower() for existing_class in data['names']]:  # Check for duplicate names
                    data['names'].append(new_class)  # Add the new class to the list of names
                    data['nc'] = len(set([name.lower() for name in data['names']]))  # Update the count of classes

                    with open(data_yaml_path, 'w') as file:  # Save the updated YAML content
                        yaml.dump(data, file)
                    print(f"Class '{new_class}' added to {data_yaml_path}")

                else:
                    print(f"Class '{new_class}' already exists in {data_yaml_path}")

            return data['names']

        # Main script
        for pill_folder in os.listdir(input_root_folder):  # iterate through the image folders
            pill_folder_path = os.path.join(input_root_folder, pill_folder)
            if os.path.isdir(pill_folder_path):  # Check if the folder is a directory

                if pill_folder != augmented_folder:  # Augmented_Photos folder is for storing the augmented uploaded images
                    new_class_name = pill_folder  # Use the name of the image folder as the class name
                    class_names = update_data_yaml(data_yaml_path, new_class_name)  # Update YAML file
                    one_hot_encoding = {class_name: index for index, class_name in enumerate(class_names)}  # Perform one-hot encoding
                
                    augmented_folder_path = os.path.join(pill_folder_path, augmented_folder)  # Path to the augmented images
                    
                    for pic_name in os.listdir(augmented_folder_path):  # Iterate through files in the augmented subfolder
                        
                        if pic_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
                            img_path = os.path.join(augmented_folder_path, pic_name)  # Adjust the path to the augmented images
                            
                            img = cv2.imread(img_path)  # Load the image
                            original = img.copy()
                            
                            # Check for existing files with the same name
                            existing_count = 0
                            base_name = os.path.splitext(pic_name)[0]  # Remove the extension
                            while True:
                                annotated_txt_path = os.path.join(output_root_folder, pill_folder, f'{base_name}{existing_count}.txt')
                                annotated_img_path = os.path.join(output_root_folder, pill_folder, f'{base_name}{existing_count}.jpeg')

                                if not os.path.exists(os.path.dirname(annotated_txt_path)):
                                    os.makedirs(os.path.dirname(annotated_txt_path))

                                if not os.path.exists(os.path.dirname(annotated_img_path)):
                                    os.makedirs(os.path.dirname(annotated_img_path))

                                if not os.path.exists(annotated_txt_path) and not os.path.exists(annotated_img_path):
                                    break

                                existing_count += 1

                            results = yolo(img)  # Perform pill detection and label

                            if results:
                                for r in results:
                                    if len(r.boxes.xywhn.tolist()) > 0:
                                        cv2.imwrite(annotated_img_path, original)
                                        x, y, w, h = r.boxes.xywhn.tolist()[0]

                                        # Save the normalized center x, center y, width, and height value.
                                        # This will be used in the training of the model
                                        with open(annotated_txt_path, 'w') as file:
                                            line = f"{one_hot_encoding[pill_folder]} {x} {y} {w} {h}\n"
                                            file.write(line)

        # Set the paths for the original and augmented folders
        annotated_folder = "./Annotated/"
        merged_folder = "./Merged_Images/"

        # Create the merged folder if it doesn't exist
        if not os.path.exists(merged_folder):
            os.makedirs(merged_folder)

        # Function to copy images and their corresponding txt files
        def copy_images_and_txt(src_folder, dest_folder):
            for class_folder in os.listdir(src_folder):
                class_path = os.path.join(src_folder, class_folder)
                if os.path.isdir(class_path):
                    for root, dirs, files in os.walk(class_path):
                        for filename in files:
                            # Check if the file is an image (png, jpg, jpeg, etc.)
                            if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                                image_path = os.path.join(root, filename)
                                txt_path = os.path.join(root, filename.rsplit(".", 1)[0] + ".txt")

                                # Copy the image to the merged folder
                                shutil.copy(image_path, dest_folder)

                                # Copy the corresponding txt file if it exists
                                if os.path.exists(txt_path):
                                    shutil.copy(txt_path, dest_folder)

        # Copy annotated images and txt files
        copy_images_and_txt(annotated_folder, merged_folder)

        # Create a Pandas DataFrame to store the data
        data = []

        # Iterate through the merged folder and read txt files
        for root, dirs, files in os.walk(merged_folder):
            for filename in files:
                if filename.lower().endswith('.txt'):
                    txt_path = os.path.join(root, filename)
                    with open(txt_path, 'r') as txt_file:
                        content = txt_file.read().split()
                        file_name = os.path.splitext(filename)[0]
                        class_label = int(content[0])  # Assuming the class is the first value in the txt file
                        data.append([file_name, class_label])

        # Create a DataFrame
        df = pd.DataFrame(data, columns=['Original_File', 'Class'])
    
        # Set the paths for the folders to store train, test, and val data
        train_folder = "./Merged_Images/train/"
        test_folder = "./Merged_Images/test/"
        val_folder = "./Merged_Images/val/"

        # Create train, test, and val folders if they don't exist
        for folder in [train_folder, test_folder, val_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                os.makedirs(os.path.join(folder, "images"))
                os.makedirs(os.path.join(folder, "labels"))

        # Use train_test_split with stratify to maintain class distribution
        train_df, test_val_df = train_test_split(df, test_size=0.3, stratify=df['Class'], random_state=42)
        test_df, val_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df['Class'], random_state=42)

        def move_files(df, destination_folder):
            for _, row in df.iterrows():
                original_file = row['Original_File']
                image_path = os.path.join(merged_folder, original_file + '.jpeg')
                txt_path = os.path.join(merged_folder, original_file + '.txt')

                # Move images to the destination folder
                shutil.move(image_path, os.path.join(destination_folder, "images"))

                # Move corresponding txt files to the destination folder
                shutil.move(txt_path, os.path.join(destination_folder, "labels"))

        # Move images and labels for the train set
        move_files(train_df, train_folder)

        # Move images and labels for the test set
        move_files(test_df, test_folder)

        # Move images and labels for the validation set
        move_files(val_df, val_folder)

        # !yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=50 imgsz=640 batch=16

        model = YOLO("yolov8s.pt")
        model.train(data='data.yaml', epochs=40, imgsz=640, batch=8)
        initialize_model()

        global training
        training = False

        return redirect(url_for('configure_pills'))
    
    except Exception as e:
        # Handle exceptions if necessary
        return f"An error occurred: {str(e)}"


@app.route('/backlog')
def backlog():
    # Fetch backlog data from the database
    backlog_data = BacklogEntry.query.all()

    # Sort the data based on the timestamp in descending order
    backlog_data.sort(key=lambda x: x.timestamp, reverse=True)

    # Filter duplicate entries based on the 'id' field
    unique_ids = set()
    filtered_backlog_data = []

    for entry in backlog_data:
        if entry.id not in unique_ids:
            print('entry id not in unique id')
            unique_ids.add(entry.id)
            filtered_backlog_data.append(entry)

    for entry in filtered_backlog_data:
        entry.screenshot_path = f'screenshot_{entry.timestamp.strftime("%Y%m%d%H%M%S")}.jpg'
        print('entry here',entry.screenshot_path)

    return render_template('fall_detection/backlog.html', backlog_data=filtered_backlog_data, datetime=datetime, url_for=url_for)


@app.route('/caregiver', methods=['POST', 'GET'])
def caregiver():
    caregivers = Caregiver.query.all()
    if request.method=='POST':
        print("Posted")
    return render_template('fall_detection/caregiver.html', caregivers=caregivers)

def allowed_file(filename): # Allowed image extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/add_task', methods=['POST'])
def add_task():
    description = request.form['description']
    new_task = Task(description=description)
    db.session.add(new_task)
    db.session.commit()
    return redirect(url_for('index'))


@app.route('/uploads/<int:caregiver_id>/<filename>')
def uploaded_file(caregiver_id, filename):
    folder_path = app.config['UPLOAD_FOLDER']
    file_path = os.path.join(folder_path, filename)


    # Check if the file exists
    # if os.path.exists(file_path):
    #     # Display the uploaded image
    #
    #     return send_from_directory(folder_path, str(caregiver_id) + '/' + filename)
    # else:

    # Check if the uploaded image is the default image
    caregiver = Caregiver.query.get(caregiver_id)

    if caregiver:
        if caregiver.photo_path == caregiver.default_photo_path:
            return send_file(file_path, as_attachment=True)
        else:
            return send_file(file_path, as_attachment=True)
    else:
        # No local directory found, and no default image specified
        return "Image not found"


@app.route('/add_caregiver', methods=['POST'])
def add_caregiver():
    if request.method == 'POST':
        print('posted')
        name = request.form['name']
        relationship = request.form['relationship']
        number = request.form['number']
        email = request.form['email']

        new_caregiver = Caregiver(name=name, relationship=relationship, number=number, email=email)

        db.session.add(new_caregiver)
        db.session.commit()
        
        # Handle photo upload
        if 'photo' in request.files:
            photo = request.files['photo']
            print(f"Filename: {photo.filename}")
            if photo.filename != '' and allowed_file(photo.filename):
                # Set the upload folder dynamically based on the caregiver ID
                upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(new_caregiver.id))
                os.makedirs(upload_folder, exist_ok=True)

                # Save the photo to the dynamic upload folder
                photo_path = os.path.join(upload_folder, secure_filename(photo.filename))
                photo.save(photo_path)

                # Update the photo path in the caregiver object
                new_caregiver.photo_path = os.path.join(str(new_caregiver.id), secure_filename(photo.filename))
                db.session.commit()

                print(f"Photo saved to: {photo_path}")
                print(f"Relative photo path: {new_caregiver.photo_path}")
            else:
                print("Invalid file format or empty filename")

        print(f"Caregiver added: {new_caregiver.id}, {new_caregiver.name}, {new_caregiver.relationship}, {new_caregiver.number}, {new_caregiver.email}, {new_caregiver.photo_path}")

        return jsonify({"message": "Caregiver added successfully"})
    
    app.route('/delete_task/<int:id>', methods=['POST'])   

def delete_task(id):
    task = Task.query.get(id)
    if task:
        db.session.delete(task)
        db.session.commit()
        return jsonify({"message": "Task deleted successfully"})
    return jsonify({"message": "Task not found"}), 404


@app.route('/delete_caregiver/<int:id>', methods=['POST'])
def delete_caregiver(id):
    caregiver = Caregiver.query.get(id)
    if caregiver:
        # Delete the associated folder and its contents
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(caregiver.id))
        shutil.rmtree(folder_path, ignore_errors=True)

        db.session.delete(caregiver)
        db.session.commit()
        return jsonify({"message": "Caregiver deleted successfully"})
    return jsonify({"message": "Caregiver not found"}), 404

@app.route('/reset_task_auto_increment', methods=['POST'])
def reset_task_auto_increment():
    # Reset the auto-increment counter to the maximum existing ID
    db.engine.execute(text("DELETE FROM sqlite_sequence WHERE name='task'"))
    return jsonify({"message": "Auto-increment counter reset successfully"})


model = YOLO('./models/fall_detection/best_80.pt')
classNames = ["Fall"]


def detect_objects():
    global fall_detected, confidence_value
    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img)

        for r in results:
            boxes = r.boxes

            # Check if there are bounding boxes
            if len(boxes) > 0:
                box = boxes[0]  # Use the first bounding box

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                text = f"{classNames[cls]}: {confidence}"
                org = (x1, y1 - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 1

                cv2.putText(img, text, org, font, fontScale, color, thickness)

                # Fall detection logic
                if classNames[cls] == "Fall":
                    fall_detected = True
                    confidence_value = confidence
                else:
                    fall_detected = False

        _, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/status')
def status():
    return jsonify({
        'fall_detected': fall_detected,
        'confidence_value': confidence_value,
        'display_text': display_text
    })

@app.route('/add_to_backlog', methods=['POST'])
def add_to_backlog():
    data = request.get_json()

    # Ensure the required fields are present in the data
    if 'events' in data and 'confidences' in data:
        events = data['events']
        confidences = data['confidences']

        # Loop through the provided data and add each entry individually
        for event, confidence in zip(events, confidences):
            if confidence:
                timestamp = datetime.now()
                screenshot_path = None

                # Check if it's a fall detection and take a screenshot
                if event == "Fall Detected" and confidence > 0.5:
                    screenshot_path = take_screenshot()

                new_entry = BacklogEntry(event=event, confidence=confidence, timestamp=timestamp, screenshot_path=screenshot_path)
                db.session.add(new_entry)
                db.session.commit()

        return jsonify({'success': True}), 200
    else:
        return jsonify({'success': False, 'error': 'Invalid data format'}), 400

def take_screenshot():
    global fall_detected

    # Only take a screenshot if the condition for capturing is met
    if fall_detected:
        _, img = cap.read()

        # Specify the directory to save the screenshots
        # screenshot_directory = r"C:\Users\matth\Desktop\devops2\FSP_IVideoAnalytics\Webapp\screenshots"
        screenshot_directory = "./screenshots"
        try:
            # Create the directory if it doesn't exist
            os.makedirs(screenshot_directory, exist_ok=True)

            # Use the timestamp for the screenshot file name
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            screenshot_filename = os.path.join(screenshot_directory, f"screenshot_{timestamp_str}.jpg")

            # Save the screenshot
            cv2.imwrite(screenshot_filename, img)
            print(f"Screenshot saved: {screenshot_filename}")

            # Play a sound
            mp3_file_path = os.path.join("static", "css", "sounds", "circles.mp3")
            pygame.mixer.music.load(mp3_file_path)  # Replace with the actual path to your sound file
            pygame.mixer.music.play()

            # Add a delay to allow the sound to play (adjust the duration as needed)
            time.sleep(2)  # 2 seconds delay, adjust as needed

            # Reset fall_detected to avoid taking multiple screenshots for the same event
            fall_detected = False

            return screenshot_filename

        except Exception as e:
            print(f"Error saving screenshot: {e}")

    return None


@app.route('/deleteFromBacklog/<int:id>', methods=['DELETE'])
def delete_from_backlog(id):
    entry_to_delete = BacklogEntry.query.get(id)

    if entry_to_delete:
        db.session.delete(entry_to_delete)
        db.session.commit()
        return jsonify({'success': True}), 200
    else:
        return jsonify({'success': False, 'error': 'Entry not found'}), 404


@app.route('/deleteAllFromBacklog', methods=['DELETE'])
def delete_all_from_backlog():
    try:
        BacklogEntry.query.delete()
        db.session.commit()

        return jsonify({'success': True}), 200
    except AttributeError:
        return jsonify({'success': False, 'error': 'Attribute error'}), 404
   

def get_all_emails():
    try:
        conn = sqlite3.connect('site2.db')  # Update with your actual database file name
        cursor = conn.cursor()

        # Assuming your backlog_entry table has an 'email' column
        cursor.execute("SELECT DISTINCT email FROM caregiver")
        emails = [row[0] for row in cursor.fetchall()]

        conn.close()

        return emails
    except Exception as e:
        print(f"Error getting emails from caregiver: {e}")
        return []


# Function to send email with attached screenshot
def send_email(receiver_emails, confidence, screenshot_path):
    try:
        email = "fypvi2023@gmail.com"
        subject = "Fall Detected"
        message = f"A fall has been detected with confidence: {confidence}"

        # Create a multipart message
        msg = MIMEMultipart()
        msg['From'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        # Attach the screenshot to the email
        with open(screenshot_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(screenshot_path)}')
            msg.attach(part)

        # Connect to the SMTP server and send the email to all recipients
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(email, "agnm taya ljad tlzl")

            # Combine all recipient emails into a single string separated by commas
            to_email = ', '.join(receiver_emails)

            msg['To'] = to_email
            server.sendmail(email, receiver_emails, msg.as_string())
            print(f"Email sent to {to_email}")

        return True
    except smtplib.SMTPException as e:
        print(f"SMTP Exception: {e}")
        return False
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


# Flask route to add an entry to the backlog and send email
@app.route('/addEntryAndEmail', methods=['POST'])
def add_entry_and_email():
    try:
        data = request.get_json()
        confidence = data.get('confidence')

        # Add entry to the backlog
        event = "Fall Detected"
        new_entry = BacklogEntry(event=event, confidence=confidence)
        db.session.add(new_entry)
        db.session.commit()

        # Get all emails from the caregiver table
        receiver_emails = [caregiver.email for caregiver in Caregiver.query.all()]
        print(f"Receiver Emails: {receiver_emails}")

        # Take a screenshot
        screenshot_path = take_screenshot()

        if screenshot_path is not None:
            # Send email to all recipients with attached screenshot
            send_success = send_email(receiver_emails, confidence, screenshot_path)

            if send_success:
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'error': 'Failed to send email'})

        else:
            return jsonify({'success': False, 'error': 'Failed to capture screenshot'})

    except Exception as e:
        print(f"Error adding entry and sending email: {e}")
        return jsonify({'success': False, 'error': str(e)})


# Memory Game Feature
@app.route('/setup_memory_game', methods=['GET', 'POST'])    
def setup_memory_game():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('image')
        category = request.form['category']

        # If the user does not select a file or category, return an error
        if not files or not category:
            flash('Please select at least one file and enter a category')
            return redirect(request.url)

        # Ensure the 'pill_folder' folder exists
        category_folder = os.path.join(app.config['MEMORY_FOLDER'], category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        # Save each file to the category folder
        for file in files:
            if file and allowed_file(file.filename):
                file.seek(0)
                # Use a secure filename to prevent malicious behavior
                secure_filename = os.path.join(category_folder, file.filename)
                file.save(secure_filename)
        
        flash('Files uploaded successfully')
        return redirect(request.url)

    return render_template('game_feature/setup_memory_game.html')


@app.route('/configure_game', methods=['GET', 'POST'])
def configure_game():
    app.config['PILL_FOLDER'] = MEMORY_FOLDER
    # Get a list of image folders within the PILL_FOLDER
    image_folders = [folder for folder in os.listdir(app.config['PILL_FOLDER']) if os.path.isdir(os.path.join(app.config['PILL_FOLDER'], folder))]
    
    # Create a dictionary to store image names for each folder
    image_data = {}
    for folder in image_folders:
        folder_path = os.path.join(app.config['PILL_FOLDER'], folder)
        image_data[folder] = [image_name for image_name in os.listdir(folder_path) if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', 'jfif'))]
    
    return render_template('pill_detection/configure_pills.html', image_data=image_data, name='Faces')

@app.route('/view_folder/<folder>')
def view_folder(folder):
    folder_path = os.path.join(app.config['PILL_FOLDER'], folder)
    images = [image_name for image_name in os.listdir(folder_path) if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', 'jfif'))]
    return render_template('pill_detection/view_folder.html', folder=folder, images=images)

@app.route('/view_image/<folder>/<image>')
def view_image(folder, image):
    folder_path = os.path.join(app.config['PILL_FOLDER'], folder)
    return send_from_directory(folder_path, image)

@app.route('/delete_folder/<folder>', methods=['POST'])
def delete_folder(folder):
    # Add logic to delete the specified folder and its contents
    folder_path = os.path.join(app.config['PILL_FOLDER'], folder)
    shutil.rmtree(folder_path)  # Use shutil.rmtree to delete a folder and its contents
    flash(f'Folder "{folder}" has been deleted successfully', 'success')

    if app.config['PILL_FOLDER'] == MEMORY_FOLDER:
        return redirect(url_for('configure_game'))
    else:
        return redirect(url_for('configure_pills'))

@app.route('/delete_image/<folder>/<image>', methods=['POST'])
def delete_image(folder, image):
    # Add logic to delete the specified image
    image_path = os.path.join(app.config['PILL_FOLDER'], folder, image)
    os.remove(image_path)  # Use os.remove to delete a single file
    flash(f'Image "{image}" has been deleted successfully', 'success')
    return redirect(url_for('view_folder', folder=folder))

@app.route('/play_memory_game', methods=['GET', 'POST'])
def play_memory_game():
    app.config['PILL_FOLDER'] = MEMORY_FOLDER
    # Get a list of image folders within the PILL_FOLDER
    image_folders = [folder for folder in os.listdir(app.config['PILL_FOLDER']) if os.path.isdir(os.path.join(app.config['PILL_FOLDER'], folder))]
    
    # Create a dictionary to store image names for each folder
    image_data = {}
    for folder in image_folders:
        folder_path = os.path.join(app.config['PILL_FOLDER'], folder)
        image_data[folder] = [image_name for image_name in os.listdir(folder_path) if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.jfif'))]
    
    return render_template('game_feature/game_template.html', image_data=image_data, name='Faces')

@app.route('/process_game_result', methods=['POST'])
def process_game_result():
    try_counter = request.form.get('try_counter')
    photo_count = request.form.get('photo_count')

    # Set entry time to the current time
    entry_time = datetime.now()

    game_result = GameResult(entry=entry_time, tries=try_counter, photo_count=photo_count)
    db.session.add(game_result)
    db.session.commit()

    # Redirect to a thank you page or the play_memory_game page
    return render_template('game_feature/finish_game.html', try_counter=try_counter, photo_count=photo_count)


@app.route('/all_results')
def all_results():
    results = GameResult.query.all()
    return render_template('game_feature/all_results.html', results=results)

@app.route('/delete_result/<int:result_id>', methods=['POST'])
def delete_result(result_id):
    # Get the GameResult record to delete
    result = GameResult.query.get_or_404(result_id)

    # Delete the record from the database
    db.session.delete(result)
    db.session.commit()

    # Redirect to the all_results page after deletion
    return redirect(url_for('all_results'))

@app.route('/analytics')
def analytics():
    # Fetch analytics data from the database (modify this based on your actual analytics data structure)
    analytics_data = BacklogEntry.query.all()

    # Extract relevant data for the charts
    timestamps = [entry.timestamp for entry in analytics_data]
    confidence_values = [entry.confidence for entry in analytics_data]
    event_names = [entry.event for entry in analytics_data]

    # Count the occurrences of each event for the bar chart
    event_counts = {event: event_names.count(event) for event in set(event_names)}

    return render_template('./fall_detection/analytics.html', timestamps=timestamps, confidence_values=confidence_values,
                           event_names=list(event_counts.keys()), event_counts=list(event_counts.values()))




@app.route('/memory_game')
def memory_game():
    return render_template('./memory_card/mgame.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Create the database tables within the Flask application context
    app.run(debug=True, port = 3000)
