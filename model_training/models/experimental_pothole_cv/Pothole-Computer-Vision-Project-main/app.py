from flask import Flask, render_template, Response, jsonify
import cv2
import supervision as sv
from ultralytics import YOLO
import time
import pyresearch 

# Flask App Initialization
app = Flask(__name__)

# PyResearch Configuration Constants
PR_MODEL_PATH = "best.pt"
PR_DISPLAY_CONFIG = {
    'window_title': "PyResearch - Pothole Computer Vision Project",
    'window_size': (1280, 720),
    'color_scheme': "PR_DARK_BLUE",
    'fps_display': True
}

# Global variable to store detection count
detection_count = 0

class PyResearchVisualizer:
    """PyResearch Standard Visualization Engine"""
    
    def __init__(self):
        self.model = YOLO(PR_MODEL_PATH)
        self.box_annotator = sv.BoundingBoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex("#0055FF")
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.7,
            text_thickness=1,
            text_color=sv.Color.WHITE,
            text_padding=10
        )
        
    def process_frame(self, frame):
        """PyResearch Standard Processing Pipeline"""
        global detection_count
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Update detection count
        detection_count = len(detections)  # Count the number of detections in the current frame
        
        # Apply PyResearch Visualization Standards
        annotated_frame = self.box_annotator.annotate(
            scene=frame,
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        return annotated_frame

def generate_frames():
    visualizer = PyResearchVisualizer()
    cap = cv2.VideoCapture("demo.mp4")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        output_frame = visualizer.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_count')
def get_detection_count():
    return jsonify({'detections': detection_count})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')