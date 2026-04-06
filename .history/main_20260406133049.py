from flask import Flask, request, render_template, Response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# load model 1 lần khi chạy app
MODEL_PATH = 'violence_model.keras'   # đổi đúng tên file model của bạn
model = load_model(MODEL_PATH, compile=False)

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 10
CLASS_LIST = ["Violence", "Nonviolence"]   # đổi thứ tự nếu label train của bạn ngược lại


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return 'Không có file', 400

    f = request.files['file']
    if f.filename == '':
        return 'Chưa chọn file', 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)

    return jsonify({
        'filename': filename,
        'filepath': filepath
    })


@app.route('/preview/<filename>')
def preview(filename):
    return render_template('preview.html', filename=filename)


@app.route('/predict/<filename>')
def predict_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(
        generate_frames(filepath),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/camera_feed')
def camera_feed():
    return Response(
        generate_frames(0),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def generate_frames(video_path):
    video_reader = cv2.VideoCapture(video_path)
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    predicted_class_name = ''
    predicted_confidence = 0.0

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame.astype("float32") / 255.0
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            input_data = np.array(frames_queue, dtype=np.float32)
            input_data = np.expand_dims(input_data, axis=0)   # (1, 10, 64, 64, 3)

            probs = model.predict(input_data, verbose=0)[0]
            predicted_label = int(np.argmax(probs))
            predicted_class_name = CLASS_LIST[predicted_label]
            predicted_confidence = float(probs[predicted_label])

        text = f'{predicted_class_name}: {predicted_confidence:.2f}'
        color = (0, 0, 255) if predicted_class_name == "Violence" else (0, 255, 0)

        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'
        )

    video_reader.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)