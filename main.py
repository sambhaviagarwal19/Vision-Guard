import cv2
import imutils
import numpy as np
from flask import Flask, render_template
import time
import threading
from threading import Thread
from gtts import gTTS
import os
import pygame

app = Flask(__name__)

# Initialize pygame for audio
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)

# Initialize detectors for both people and objects
hog_people = cv2.HOGDescriptor()
hog_objects = cv2.HOGDescriptor()

# Set up people detector
hog_people.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Set up object detector using a pre-trained cascade classifier
objects_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_upperbody.xml'
)

# Camera parameters for distance estimation
KNOWN_OBJECT_HEIGHT = 0.5  # Average object height in meters
KNOWN_PERSON_HEIGHT = 1.7  # Average person height in meters
FOCAL_LENGTH = 600        # Approximate focal length in pixels
MIN_SAFE_DISTANCE = 4.2   # Minimum safe distance in meters
TOO_CLOSE_DISTANCE = 4.2  # Distance considered dangerously close
GUIDANCE_REPEAT_INTERVAL = 2  # Repeat guidance every 2 seconds

# Audio feedback variables
last_audio_time = 0
current_guidance = ""
temp_audio_files = []
last_guidance_time = 0

class MessageManager:
    def __init__(self):
        self.message_lock = threading.Lock()
        self.current_message = None
        self.message_completed = threading.Event()
        self.message_completed.set()
        self.is_movement_message = False
        
    def text_to_speech(self, text, lang='en', is_movement=False):
        """Convert text to speech and play it"""
        try:
            tts = gTTS(text=text, lang=lang)
            filename = f"temp_{int(time.time())}.mp3"
            tts.save(filename)
            temp_audio_files.append(filename)
            
            # Wait for previous message to complete only if it's not a movement message
            if not is_movement and not self.is_movement_message:
                self.message_completed.wait()
            
            # Acquire lock to prevent other messages from playing
            with self.message_lock:
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                self.current_message = filename
                self.is_movement_message = is_movement
                self.message_completed.clear()
                
                # Wait for the audio to finish playing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                self.message_completed.set()
                self.current_message = None
                self.is_movement_message = False
                
        except Exception as e:
            print(f"Error in text_to_speech: {e}")
            self.message_completed.set()

message_manager = MessageManager()

def provide_continuous_feedback(regions, frame_width, is_people):
    """Generate continuous audio feedback with repeated guidance"""
    global last_audio_time, current_guidance, last_guidance_time
    current_time = time.time()
    
    guidance, distance, changed = get_guidance_direction(regions, frame_width, is_people)
    
    # Only generate new audio if guidance has changed or it's time for a repeat
    if changed or current_time - last_guidance_time >= GUIDANCE_REPEAT_INTERVAL:
        current_guidance = guidance
        if regions is not None and len(regions) > 0:
            closest_person = min([estimate_distance(h, is_person) for (x, y, w, h, is_person) in regions])
            
            if closest_person < TOO_CLOSE_DISTANCE:
                urgency = "Warning! Too close! "
            elif closest_person < MIN_SAFE_DISTANCE:
                urgency = "Caution! "
            else:
                urgency = ""
                
            subject = "person" if is_people else "object"
            subjects = "people" if is_people else "objects"
            
            # Create comprehensive message
            message = f"{urgency}{len(regions)} {subject}{'s' if len(regions) != 1 else ''} detected. "
            
            # Handle movement messages separately
            if guidance == "move left" or guidance == "move right":
                Thread(target=message_manager.text_to_speech, 
                      args=(message + f"Move {guidance}.",), 
                      kwargs={'is_movement': True}).start()
            else:
                Thread(target=message_manager.text_to_speech, 
                      args=(message + f"Closest is {closest_person:.1f} meters away",)).start()
            
            last_guidance_time = current_time

def clean_temp_files():
    """Clean up temporary audio files"""
    for file in temp_audio_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass
    temp_audio_files.clear()

def estimate_distance(pixel_height, is_person=False):
    """Estimate distance to an object based on its pixel height"""
    if pixel_height == 0:
        return float('inf')
    known_height = KNOWN_PERSON_HEIGHT if is_person else KNOWN_OBJECT_HEIGHT
    return (known_height * FOCAL_LENGTH) / pixel_height

def get_detections(frame):
    """Get detections for both people and objects"""
    # Detect people using HOG
    people_regions, weights = hog_people.detectMultiScale(
        frame, 
        winStride=(4, 4), 
        padding=(8, 8), 
        scale=1.05
    )
    
    # Detect objects using Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = objects_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Initialize the combined regions list
    combined_regions = []  # Add this line
    
    # Combine regions and track what type of detection each is
    for region in people_regions:
        combined_regions.append((*region, True))  # True indicates person
    for region in objects:
        combined_regions.append((*region, False))  # False indicates object
        
    return people_regions, objects

def get_guidance_direction(regions, frame_width, is_people):
    """Determine if user should move left or right to avoid obstacles"""
    if regions is None or len(regions) == 0:
        return "path clear", float('inf'), False
    
    obstacles = []
    for region in regions:  # Changed from (x, y, w, h) to just region
        x, y, w, h, is_person = region  # Unpack all 5 values
        center_x = x + w//2
        distance = estimate_distance(h, is_person)
        obstacles.append((center_x, distance))
    
    # Divide frame into left, center, and right zones
    left_threshold = frame_width // 3
    right_threshold = 2 * frame_width // 3
    
    # Find all obstacles in each zone
    left_obstacles = [(cx, dist) for (cx, dist) in obstacles if cx < left_threshold]
    center_obstacles = [(cx, dist) for (cx, dist) in obstacles if left_threshold <= cx <= right_threshold]
    right_obstacles = [(cx, dist) for (cx, dist) in obstacles if cx > right_threshold]
    
    # Find closest obstacle in each zone
    closest_left = min(left_obstacles, key=lambda x: x[1], default=(0, float('inf')))
    closest_center = min(center_obstacles, key=lambda x: x[1], default=(0, float('inf')))
    closest_right = min(right_obstacles, key=lambda x: x[1], default=(0, float('inf')))
    
    changed = False
    guidance = "path clear"
    closest_dist = float('inf')
    
    if closest_center[1] < TOO_CLOSE_DISTANCE:
        if closest_left[1] < closest_right[1]:
            guidance = "move right"
        else:
            guidance = "move left"
        closest_dist = closest_center[1]
        changed = True
        
    elif closest_center[1] < MIN_SAFE_DISTANCE:
        if len(left_obstacles) == 0 and len(right_obstacles) > 0:
            guidance = "move left"
        elif len(right_obstacles) == 0 and len(left_obstacles) > 0:
            guidance = "move right"
        elif closest_left[1] < closest_right[1]:
            guidance = "move right"
        else:
            guidance = "move left"
        closest_dist = closest_center[1]
        changed = True
            
    elif closest_left[1] < MIN_SAFE_DISTANCE and closest_right[1] >= MIN_SAFE_DISTANCE:
        guidance = "move right"
        closest_dist = closest_left[1]
        changed = True
            
    elif closest_right[1] < MIN_SAFE_DISTANCE and closest_left[1] >= MIN_SAFE_DISTANCE:
        guidance = "move left"
        closest_dist = closest_right[1]
        changed = True
            
    elif closest_left[1] < MIN_SAFE_DISTANCE and closest_right[1] < MIN_SAFE_DISTANCE:
        if closest_left[1] < closest_right[1]:
            guidance = "move right"
        else:
            guidance = "move left"
        closest_dist = min(closest_left[1], closest_right[1])
        changed = True
    
    return guidance, closest_dist, changed

@app.route("/")
@app.route("/home")
def home():
    clean_temp_files()
    return render_template("home.html")

@app.route("/webcam")
def webcam():
    clean_temp_files()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Webcam not available", 500
    
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
            
        image = imutils.resize(image, width=min(500, image.shape[1]))
        
        # Get both people and object detections
        people_regions, objects = get_detections(image)
        
        # Combine regions and track what type of detection each is
        combined_regions = []
        for region in people_regions:
            combined_regions.append((*region, True))  # True indicates person
        for region in objects:
            combined_regions.append((*region, False))  # False indicates object
        
        # Provide feedback based on all detections
        provide_continuous_feedback(combined_regions, image.shape[1], None)
        
        # Draw rectangles around all detections
        for x, y, w, h, is_person in combined_regions:
            distance = estimate_distance(h, is_person)
            
            # Choose color based on distance
            color = (
                (0, 0, 255) if distance < TOO_CLOSE_DISTANCE 
                else (0, 255, 0) if distance < MIN_SAFE_DISTANCE 
                else (255, 0, 0)
            )
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                image, f"{distance:.1f}m", 
                (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                1
            )
        
        # Add text overlays showing counts
        guidance, _, changed = get_guidance_direction(combined_regions, image.shape[1], None)
        cv2.putText(
            image, f"People: {len(people_regions)}", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        cv2.putText(
            image, f"Objects: {len(objects)}", 
            (20, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        cv2.putText(
            image, f"Guidance: {guidance}", 
            (20, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        cv2.imshow("Detection from Webcam", image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return render_template("webcam.html")

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    finally:
        clean_temp_files()