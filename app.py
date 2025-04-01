from flask import Flask, render_template, request, send_from_directory, redirect, url_for, Response, jsonify
import cv2 as cv
import numpy as np
import os
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# YOLO setup
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Global variables for progress tracking
progress = {
    'total_frames': 0,
    'processed_frames': 0
}

# Configuration parameters
focal_length = 1109
car_real_height = 1.5  # meters
pixels_per_meter = 72
fps = 30
safety_time = 3.0  # seconds
safety_distance = 50  # Updated to 50 meters for caution message

# Optical flow params
prev_gray = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_gradient_circle(img, center, size):
    """Draw gradient circle around detected vehicles"""
    for i in range(size, 0, -1):
        intensity = int(255 * (i / size))
        cv.circle(img, center, i, (255 - intensity, 0, intensity), -1)

def calculate_curvature(left_line, right_line):
    if not left_line or not right_line:
        return 0.0

    def line_angle(line):
        (x1, y1), (x2, y2) = line
        if x2 == x1:
            return 90.0
        return np.degrees(np.arctan((y2 - y1) / (x2 - x1)))

    left_angle = line_angle(left_line)
    right_angle = line_angle(right_line)
    return (abs(left_angle) + abs(right_angle)) / 2

def region_of_interest(image):
    height, width = image.shape[:2]
    top_y = int(height * 0.7)
    polygon = np.array([[(0, height), (width // 2, top_y), (width, height)]], np.int32)
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygon, 255)
    return cv.bitwise_and(image, mask)

def detect_lanes(image, prev_left=None, prev_right=None):
    height, width = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    roi_edges = region_of_interest(edges)

    lines = cv.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)

    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0: continue
            slope = dy / dx
            if abs(slope) < 0.5: continue
            (left_lines if slope < 0 else right_lines).append(line[0])

    def process_lines(lines, prev_line):
        if not lines: return prev_line
        points = np.array([p for line in lines for p in [line[:2], line[2:]]])
        if len(points) < 2: return prev_line
        m, b = np.polyfit(points[:, 1], points[:, 0], 1)
        top_y = int(height * 0.7)
        return [(int(m * height + b), height), (int(m * top_y + b), top_y)]

    left_line = process_lines(left_lines, prev_left)
    right_line = process_lines(right_lines, prev_right)

    lane_image = np.copy(image)
    if left_line and right_line:
        pts = np.array([left_line[0], left_line[1], right_line[1], right_line[0]], np.int32)
        cv.fillPoly(lane_image, [pts], (0, 255, 0))
        cv.polylines(lane_image, [np.array(left_line, np.int32)], False, (255, 200, 0), 3)
        cv.polylines(lane_image, [np.array(right_line, np.int32)], False, (255, 200, 0), 3)

    return lane_image, left_line, right_line

def detect_vehicles(frame, left_line, right_line):
    height, width = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    if blob is None:
        print("Error: Failed to create blob from image.")
        return []
    net.setInput(blob)
    try:
        outs = net.forward(output_layers)
    except Exception as e:
        print(f"Error during YOLO forward pass: {e}")
        return []

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ['car', 'truck', 'bus']:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    vehicles = []

    if len(indexes) > 0:
        for i in indexes.flatten():
            box = boxes[i]
            x, y, w, h = box
            distance = (car_real_height * focal_length) / h if h != 0 else 0
            center = (x + w // 2, y + h // 2)

            lane = 'current'
            if left_line and right_line:
                x_left = np.interp(center[1],
                                 [left_line[0][1], left_line[1][1]],
                                 [left_line[0][0], left_line[1][0]])
                x_right = np.interp(center[1],
                                  [right_line[0][1], right_line[1][1]],
                                  [right_line[0][0], right_line[1][0]])

                if center[0] < x_left:
                    lane = 'left'
                elif center[0] > x_right:
                    lane = 'right'

            vehicles.append((box, distance, lane))

    return vehicles

def generate_frames(input_path, output_path):
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    global prev_gray, progress
    prev_left = None
    prev_right = None

    # Get total frames
    progress['total_frames'] = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    progress['processed_frames'] = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Increment processed frames
        progress['processed_frames'] += 1

        # Lane detection
        lane_frame, left_line, right_line = detect_lanes(frame, prev_left, prev_right)
        prev_left = left_line
        prev_right = right_line

        # Vehicle detection
        vehicles = detect_vehicles(frame, left_line, right_line)

        # Speed calculation using optical flow
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        speed_kmh = 0
        if prev_gray is not None:
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
            if magnitude.size > 0:
                avg_speed_pixels = np.mean(magnitude[-100:, :])
                speed_mps = (avg_speed_pixels * fps) / pixels_per_meter
                speed_kmh = speed_mps * 3.6
        prev_gray = gray

        # Safety parameters
        min_distances = {'current': float('inf'), 'left': float('inf'), 'right': float('inf')}
        current_lane_clear = True
        suggestion = None
        curvature = calculate_curvature(left_line, right_line)

        # Process detected vehicles
        for (box, distance, lane) in vehicles:
            x, y, w, h = box
            center = (x + w // 2, y + h // 2)

            # Update minimum distances
            min_distances[lane] = min(min_distances[lane], distance)

            # Check current lane obstruction
            if lane == 'current' and distance < safety_distance:
                current_lane_clear = False

            # Draw vehicle markers
            create_gradient_circle(lane_frame, center, 20)
            cv.putText(lane_frame, f"{distance:.1f}m", (x, y - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Determine lane status and suggestions
        lane_status = 'clear'
        ttc = min_distances['current'] / (speed_kmh / 3.6 + 1e-5)  # Time-to-collision

        if not current_lane_clear:
            lane_status = 'blocked'
            if min_distances['left'] > safety_distance:
                suggestion = 'left'
            elif min_distances['right'] > safety_distance:
                suggestion = 'right'

        # Draw overlays
        # Primary info box (top-left)
        cv.rectangle(lane_frame, (0, 0), (350, 160), (0, 0, 0), -1)
        status_color = (0, 255, 0) if lane_status == 'clear' else (0, 0, 255)
        status_text = "CLEAR PATH" if lane_status == 'clear' else f"CAUTION: Vehicle {min_distances['current']:.1f}m Ahead!"
        cv.putText(lane_frame, status_text, (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv.putText(lane_frame, f"Speed: {speed_kmh:.1f} km/h", (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv.putText(lane_frame, f"Curvature: {curvature:.1f}°", (10, 90),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv.putText(lane_frame, f"TTC: {ttc:.1f}s", (10, 120),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Secondary info box (top-left below primary)
        cv.rectangle(lane_frame, (0, 170), (350, 300), (0, 0, 0), -1)
        cv.putText(lane_frame, "Adjacent Vehicles:", (10, 200),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv.putText(lane_frame, f"Front: {min_distances['current']:.1f}m", (10, 230),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv.putText(lane_frame, f"Left: {min_distances['left']:.1f}m", (10, 260),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv.putText(lane_frame, f"Right: {min_distances['right']:.1f}m", (10, 290),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Draw suggestion arrow
        if suggestion:
            cv.putText(lane_frame, f"Suggested Lane: {suggestion.upper()}",
                      (lane_frame.shape[1] - 300, 50),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Write the frame into the output video file
        out.write(lane_frame)

        # Encode the frame as JPEG
        ret, buffer = cv.imencode('.jpg', lane_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return Response(generate_frames(input_path, output_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/progress/<filename>')
def get_progress(filename):
    global progress
    if progress['total_frames'] == 0:
        return jsonify({'progress': 0})
    progress_percent = (progress['processed_frames'] / progress['total_frames']) * 100
    return jsonify({'progress': progress_percent})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            return render_template('index.html', filename=filename)
    return render_template('index.html', filename=None)  # Pass filename=None for GET requests

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)