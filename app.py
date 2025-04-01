import argparse
import time
import queue
import threading

import numpy as np
import cv2 as cv
import joblib
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

YUNET_MODEL = "model/face_detection_yunet_2023mar.onnx"
SFACENET_MODEL = "model/face_recognition_sface_2021dec.onnx"

svc = joblib.load('model/svc.pkl')
mydict = ['Bao', 'Chi Thanh', 'Huu Loc', 'Linh Phan', 'Thai Hung']


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError


parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str,
                    help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default=YUNET_MODEL,
                    help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default=SFACENET_MODEL,
                    help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9,
                    help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False,
                    help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args([])  # Empty list to avoid conflicts with Streamlit


def visualize(input, faces, fps, thickness=2, skip_recognition=False):
    output = input.copy()
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            x, y, w, h = coords[0], coords[1], coords[2], coords[3]

            # Draw bounding box
            cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), thickness)

            if not skip_recognition:
                try:
                    # Align & extract features - potentially heavy operation
                    face_align = recognizer.alignCrop(input, face)
                    face_feature = recognizer.feature(face_align)
                    prediction = svc.predict(face_feature)[0]
                    label = mydict[prediction]
                    
                    # Draw name label above the box
                    cv.putText(output, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    # Handle any errors in recognition gracefully
                    cv.putText(output, "Unknown", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw landmarks - only if needed
            if st.session_state.get('show_landmarks', True):
                for i in range(5):
                    cx, cy = coords[4 + i * 2], coords[5 + i * 2]
                    cv.circle(output, (cx, cy), 2, (255, 0, 255), thickness)

    cv.putText(output, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output


class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize shared variables
        if 'frame_count' not in st.session_state:
            st.session_state['frame_count'] = 0
        if 'last_recognition_time' not in st.session_state:
            st.session_state['last_recognition_time'] = 0
        
        self.detector = cv.FaceDetectorYN.create(
            args.face_detection_model,
            "",
            (320, 320),
            args.score_threshold,
            args.nms_threshold,
            args.top_k
        )
        self.recognizer = cv.FaceRecognizerSF.create(
            args.face_recognition_model, "")
        self.last_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.skip_count = 0
        self.resolution_scale = st.session_state.get('resolution_scale', 1.0)
        self.recognition_interval = st.session_state.get('recognition_interval', 0.5)  # seconds between recognition
        
        # Create result queue for thread-safe result passing
        self.result_queue = queue.Queue(maxsize=1)
        self.frame_queue = queue.Queue(maxsize=5)  # Buffer a few frames
        self.processing_thread = None
        self.processing = True
        
        # Start background processing thread
        self.start_processing_thread()

    def start_processing_thread(self):
        """Start a separate thread for face detection/recognition processing"""
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True  # Thread will close when main program exits
        self.processing_thread.start()
        
    def process_frames(self):
        """Process frames in background thread"""
        while self.processing:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)  # Small sleep to avoid CPU spinning
                    continue
                
                img = self.frame_queue.get()
                if img is None:
                    continue
                    
                # Downsample the image if needed to improve performance
                if self.resolution_scale < 1.0:
                    h, w = img.shape[:2]
                    new_h, new_w = int(h * self.resolution_scale), int(w * self.resolution_scale)
                    img_small = cv.resize(img, (new_w, new_h))
                    
                    # Update detector input size
                    self.detector.setInputSize([new_w, new_h])
                    
                    # Detect faces on smaller image (faster)
                    faces = self.detector.detect(img_small)
                    
                    # Scale coordinates back to original size
                    if faces[1] is not None:
                        scale_factor = 1.0 / self.resolution_scale
                        for i in range(len(faces[1])):
                            # Scale coordinates (x,y,w,h and landmarks)
                            faces[1][i][0:4] *= scale_factor  # x, y, w, h
                            for j in range(5):  # 5 landmarks
                                faces[1][i][4+j*2] *= scale_factor  # x coord
                                faces[1][i][5+j*2] *= scale_factor  # y coord
                else:
                    # Process at full resolution
                    h, w = img.shape[:2]
                    self.detector.setInputSize([w, h])
                    faces = self.detector.detect(img)
                
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 30.0
                self.last_time = current_time
                
                # Only do recognition at intervals to save processing power
                skip_recognition = (current_time - st.session_state['last_recognition_time']) < self.recognition_interval
                
                if not skip_recognition:
                    st.session_state['last_recognition_time'] = current_time
                
                # Process and visualize results
                result = visualize(img, faces, fps, skip_recognition=skip_recognition)
                
                # Update the result queue (remove old result if queue is full)
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put(result)
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                time.sleep(0.1)

    def recv(self, frame):
        """Receive frames from WebRTC and pass to processing thread"""
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Add frame to processing queue (non-blocking)
            if not self.frame_queue.full():
                self.frame_queue.put(img)
            
            # Use the most recent result or the current frame if no result available
            try:
                result_img = self.result_queue.get_nowait()
            except queue.Empty:
                # If no processed result is available, just return the original frame
                result_img = img
                
            # Wrap the processed image in a VideoFrame
            return av.VideoFrame.from_ndarray(result_img, format="bgr24")
            
        except Exception as e:
            st.error(f"Error in recv: {str(e)}")
            return frame
    
    def on_ended(self):
        """Clean up when the stream ends"""
        self.processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)


# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['resolution_scale'] = 0.5  # Default to half resolution for better performance
    st.session_state['recognition_interval'] = 0.5  # Default recognition every 0.5 seconds
    st.session_state['show_landmarks'] = True
    
# Streamlit UI
st.title("Face Recognition System")
st.write("This application detects and recognizes faces in real-time.")

# Optional sidebar controls
with st.sidebar:
    st.header("Performance Settings")
    
    # Resolution control
    resolution_scale = st.slider("Resolution Scale", 0.25, 1.0, 
                              st.session_state.get('resolution_scale', 0.5), 0.05,
                              help="Lower values improve performance but reduce quality")
    st.session_state['resolution_scale'] = resolution_scale
    
    # Recognition frequency control
    recognition_interval = st.slider("Recognition Interval (seconds)", 0.1, 2.0, 
                                  st.session_state.get('recognition_interval', 0.5), 0.1,
                                  help="Longer intervals improve performance")
    st.session_state['recognition_interval'] = recognition_interval
    
    # Detection confidence
    score_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.9, 0.05)
    args.score_threshold = score_threshold
    
    # UI controls
    st.session_state['show_landmarks'] = st.checkbox("Show Facial Landmarks", 
                                                   st.session_state.get('show_landmarks', True))

# Initialize detector and recognizer globally
detector = cv.FaceDetectorYN.create(
    args.face_detection_model,
    "",
    (320, 320),
    args.score_threshold,
    args.nms_threshold,
    args.top_k
)
recognizer = cv.FaceRecognizerSF.create(
    args.face_recognition_model, "")

# Set up WebRTC streamer with optimized settings
webrtc_ctx = webrtc_streamer(
    key="face-recognition",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceDetectionProcessor,
    media_stream_constraints={"video": {"frameRate": {"ideal": 15}}, "audio": False},
    async_processing=True,
    video_html_attrs={"style": {"width": "100%", "height": "auto"}, "autoplay": True, "controls": True},
    frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    server_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# Display recognized people names
st.subheader("Available Recognitions:")
for name in mydict:
    st.write(f"- {name}")

# Instructions
st.markdown("""
### Instructions
1. Allow camera access when prompted
2. Look at the camera to see face detection and recognition
3. Adjust settings in the sidebar to optimize performance:
   - Lower resolution scale for smoother performance
   - Increase recognition interval to reduce CPU usage
   - Adjust detection confidence as needed
""")

# Performance tips
with st.expander("Performance Tips"):
    st.markdown("""
    - If the stream is lagging, try reducing the resolution scale
    - Increase the recognition interval for smoother performance
    - Disable facial landmarks if not needed
    - Ensure good lighting for better face detection
    - Try closing other browser tabs or applications
    """)
