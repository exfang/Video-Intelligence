import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data4.db'
db = SQLAlchemy(app)

app.config['UPLOAD_FOLDER'] = 'uploads'

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)

class Caregiver(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    relationship = db.Column(db.String(50), nullable=False)
    number = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    default_photo_path = r'C:\Users\matth\PycharmProjects\fyp\static\user.jpg'
    photo_path = db.Column(db.String(255), default=default_photo_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_task', methods=['POST'])
def add_task():
    description = request.form['description']
    new_task = Task(description=description)
    db.session.add(new_task)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/backlog')
def backlog():
    tasks = Task.query.all()
    return render_template('backlog.html', tasks=tasks)

@app.route('/caregiver', methods=['POST', 'GET'])
def caregiver():
    caregivers = Caregiver.query.all()
    if request.method=='POST':
        print("Posted")
    return render_template('caregiver.html', caregivers=caregivers)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

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

import shutil  # Add this import at the beginning of your script

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


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=3000)

