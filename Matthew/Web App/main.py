# Import necessary libraries
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Access the computer's webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode the frame into JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break

            # Convert the frame to bytes
            frame_bytes = buffer.tobytes()

            # Yield the frame in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Render the HTML template
    return render_template('index.html')

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/video_feed')
def video_feed():
    # Return the response with the video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

