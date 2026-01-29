from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

# Define important joint indices
JOINTS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28
}

# Define pose types and their important joints
POSE_IMPORTANT_JOINTS = {
    'pose1': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'],
    'pose2': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'],
    'pose3': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'],
    'pose4': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'],
    'belly': ['left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'balance': ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
    'ball': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
    'resistance': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
    'stretch_arm': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def extract_joint_angles(landmarks):
    joint_angles = {}

    for joint_name, joint_index in JOINTS.items():
        # Get the coordinates for the joints required for angle calculation
        if joint_name.endswith('elbow'):
            shoulder = landmarks[JOINTS['left_shoulder'] if 'left' in joint_name else JOINTS['right_shoulder']]
            elbow = landmarks[joint_index]
            wrist = landmarks[JOINTS['left_wrist'] if 'left' in joint_name else JOINTS['right_wrist']]
            joint_angles[joint_name] = calculate_angle(
                [shoulder.x, shoulder.y], 
                [elbow.x, elbow.y], 
                [wrist.x, wrist.y]
            )
        elif joint_name.endswith('shoulder'):
            hip = landmarks[JOINTS['left_hip'] if 'left' in joint_name else JOINTS['right_hip']]
            shoulder = landmarks[joint_index]
            elbow = landmarks[JOINTS['left_elbow'] if 'left' in joint_name else JOINTS['right_elbow']]
            joint_angles[joint_name] = calculate_angle(
                [hip.x, hip.y], 
                [shoulder.x, shoulder.y], 
                [elbow.x, elbow.y]
            )
        elif joint_name.endswith('hip'):
            shoulder = landmarks[JOINTS['left_shoulder'] if 'left' in joint_name else JOINTS['right_shoulder']]
            hip = landmarks[joint_index]
            knee = landmarks[JOINTS['left_knee'] if 'left' in joint_name else JOINTS['right_knee']]
            joint_angles[joint_name] = calculate_angle(
                [shoulder.x, shoulder.y], 
                [hip.x, hip.y], 
                [knee.x, knee.y]
            )
        elif joint_name.endswith('knee'):
            hip = landmarks[JOINTS['left_hip'] if 'left' in joint_name else JOINTS['right_hip']]
            knee = landmarks[joint_index]
            ankle = landmarks[JOINTS['left_ankle'] if 'left' in joint_name else JOINTS['right_ankle']]
            joint_angles[joint_name] = calculate_angle(
                [hip.x, hip.y], 
                [knee.x, knee.y], 
                [ankle.x, ankle.y]
            )
        elif joint_name.endswith('wrist'):
            elbow = landmarks[JOINTS['left_elbow'] if 'left' in joint_name else JOINTS['right_elbow']]
            wrist = landmarks[joint_index]
            # Use index finger as third point (for wrist angle)
            index_finger = landmarks[19 if 'left' in joint_name else 20]  # Index finger MCP joint
            joint_angles[joint_name] = calculate_angle(
                [elbow.x, elbow.y], 
                [wrist.x, wrist.y], 
                [index_finger.x, index_finger.y]
            )

    return joint_angles

# Function to extract reference angles from a reference image
def extract_reference_angles(reference_image_path, pose_type):
    # Add more detailed print statements
    print(f"Attempting to extract reference angles for {pose_type}")
    print(f"Looking for reference image at: {reference_image_path}")
    
    # Check if reference image exists
    if not os.path.exists(reference_image_path):
        print(f"ERROR: Reference image not found: {reference_image_path}")
        return {}
    
    reference_angles = {}
    angle_tolerance = 20  # Default tolerance of ±20 degrees
    
    # Check if reference image exists
    if not os.path.exists(reference_image_path):
        print(f"Reference image not found: {reference_image_path}")
        return reference_angles
    
    # Read the reference image
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print(f"Failed to read reference image: {reference_image_path}")
        return reference_angles
    
    # Process the reference image
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        model_complexity=1
    ) as pose:
        # Convert image to RGB
        reference_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        results = pose.process(reference_rgb)
        
        if results.pose_landmarks:
            # Extract joint angles from reference image
            reference_landmarks = results.pose_landmarks.landmark
            joint_angles = extract_joint_angles(reference_landmarks)
            
            # Get important joints for this pose type
            important_joints = POSE_IMPORTANT_JOINTS.get(pose_type, list(JOINTS.keys()))
            
            # Create reference data with tolerance
            for joint in important_joints:
                if joint in joint_angles:
                    angle = joint_angles[joint]
                    reference_angles[joint] = {
                        'angle': angle,
                        'min': max(0, angle - angle_tolerance),
                        'max': min(180, angle + angle_tolerance)
                    }
            
            # Draw the reference pose for debugging
            annotated_image = reference_image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec
            )
            
            # Save the annotated reference image
            output_path = os.path.join('static', 'annotated_reference_pose', f"{pose_type}_annotated.jpg")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            
            print(f"Reference angles extracted for {pose_type}: {reference_angles}")
        else:
            print(f"No pose detected in reference image: {reference_image_path}")
    
    return reference_angles

# Cache for reference angles
reference_angles_cache = {}

def get_reference_angles(pose_type):
    global reference_angles_cache
    
    # Check if we already processed this pose
    if pose_type in reference_angles_cache:
        return reference_angles_cache[pose_type]
    
    # Define reference image path
    reference_image_path = os.path.join('static', 'reference_poses', f"{pose_type}.jpg")
    
    # Default angles as fallback
    default_references = {
        'pose1': {
            'left_shoulder': {'angle': 90, 'min': 60, 'max': 130}, 
            'right_shoulder': {'angle': 90, 'min': 60, 'max': 130},
            'left_elbow': {'angle': 120, 'min': 30, 'max': 180},
            'right_elbow': {'angle': 120, 'min': 30, 'max': 180}
        },
        'pose2': {
            'left_shoulder': {'angle': 45, 'min': 25, 'max': 65},
            'right_shoulder': {'angle': 45, 'min': 25, 'max': 65},
            'left_elbow': {'angle': 120, 'min': 100, 'max': 140},
            'right_elbow': {'angle': 120, 'min': 100, 'max': 140}
        },
        'pose3': {
            'left_shoulder': {'angle': 60, 'min': 40, 'max': 80},
            'right_shoulder': {'angle': 60, 'min': 40, 'max': 80},
            'left_elbow': {'angle': 135, 'min': 115, 'max': 155},
            'right_elbow': {'angle': 135, 'min': 115, 'max': 155}
        },
        'pose4': {
            'left_shoulder': {'angle': 75, 'min': 55, 'max': 95},
            'right_shoulder': {'angle': 75, 'min': 55, 'max': 95},
            'left_elbow': {'angle': 150, 'min': 130, 'max': 170},
            'right_elbow': {'angle': 150, 'min': 130, 'max': 170}
        },
        'belly': {
            'left_hip': {'angle': 170, 'min': 150, 'max': 180},
            'right_hip': {'angle': 170, 'min': 150, 'max': 180},
            'left_knee': {'angle': 170, 'min': 150, 'max': 180},
            'right_knee': {'angle': 170, 'min': 150, 'max': 180}
        },
        'balance': {
            'left_hip': {'angle': 160, 'min': 140, 'max': 180},
            'right_hip': {'angle': 160, 'min': 140, 'max': 180},
            'left_knee': {'angle': 160, 'min': 140, 'max': 180},
            'right_knee': {'angle': 160, 'min': 140, 'max': 180},
            'left_ankle': {'angle': 90, 'min': 70, 'max': 110},
            'right_ankle': {'angle': 90, 'min': 70, 'max': 110}
        },
        'ball': {
            'left_shoulder': {'angle': 80, 'min': 60, 'max': 100},
            'right_shoulder': {'angle': 80, 'min': 60, 'max': 100},
            'left_elbow': {'angle': 100, 'min': 80, 'max': 120},
            'right_elbow': {'angle': 100, 'min': 80, 'max': 120},
            'left_wrist': {'angle': 160, 'min': 140, 'max': 180},
            'right_wrist': {'angle': 160, 'min': 140, 'max': 180}
        },
        'resistance': {
            'left_shoulder': {'angle': 70, 'min': 50, 'max': 90},
            'right_shoulder': {'angle': 70, 'min': 50, 'max': 90},
            'left_elbow': {'angle': 110, 'min': 90, 'max': 130},
            'right_elbow': {'angle': 110, 'min': 90, 'max': 130},
            'left_wrist': {'angle': 170, 'min': 150, 'max': 180},
            'right_wrist': {'angle': 170, 'min': 150, 'max': 180}
        }
    }
    
    # Try to extract reference angles from image
    reference_data = extract_reference_angles(reference_image_path, pose_type)
    
    # If no reference data extracted, use default if available
    if not reference_data and pose_type in default_references:
        reference_data = default_references[pose_type]
        print(f"Using default reference angles for {pose_type}")
    
    # Cache the reference data
    reference_angles_cache[pose_type] = reference_data
    
    return reference_data

def generate_frames(pose_type='stretch'):
    camera = cv2.VideoCapture(0)
    
    # Get important joints for the current pose
    important_joints = POSE_IMPORTANT_JOINTS.get(pose_type, list(JOINTS.keys()))
    
    # Get reference angles for the specified pose type
    reference_data = get_reference_angles(pose_type)
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # Mirror flip
            
            # Prepare output frame
            output_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract joint angles
                live_joint_angles = extract_joint_angles(landmarks)
                
                # Check if joint angles match the reference
                correct_joints = 0
                total_joints = len(important_joints)
                
                # Add debug info to see actual angles
                debug_y = 30
                
                for joint in important_joints:
                    if joint in live_joint_angles:
                        live_angle = live_joint_angles[joint]
                        
                        # Get joint coordinates for display
                        joint_index = JOINTS[joint]
                        joint_coords = (
                            int(landmarks[joint_index].x * frame.shape[1]),
                            int(landmarks[joint_index].y * frame.shape[0])
                        )
                        
                        # Default values if no reference
                        ref_min = 0
                        ref_max = 180
                        ref_angle = None
                        
                        # If we have reference data, use it
                        if reference_data and joint in reference_data:
                            ref_angle = reference_data[joint]['angle']
                            ref_min = reference_data[joint]['min']
                            ref_max = reference_data[joint]['max']
                        
                        # Display debug info for each joint
                        debug_text = f"{joint}: {int(live_angle)}°"
                        if ref_angle is not None:
                            debug_text += f" (Target: {int(ref_angle)}±20°)"
                        
                        cv2.putText(
                            output_frame, debug_text, 
                            (10, debug_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )
                        debug_y += 20
                        
                        # Determine if the angle matches the reference range
                        if reference_data and joint in reference_data and ref_min <= live_angle <= ref_max:
                            match_text = "CORRECT"
                            text_color = (0, 255, 0)  # Green for match
                            # Draw green circle around the joint
                            cv2.circle(output_frame, joint_coords, 15, (0, 255, 0), -1)
                            correct_joints += 1
                        else:
                            match_text = "ADJUST"
                            text_color = (0, 0, 255)  # Red for unmatch
                            # Draw red circle around the joint
                            cv2.circle(output_frame, joint_coords, 15, (0, 0, 255), -1)
                        
                        # Display text information near joint
                        display_text = f"{int(live_angle)}°"
                        cv2.putText(
                            output_frame, display_text, 
                            (joint_coords[0] - 15, joint_coords[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA
                        )
                
                # Draw pose landmarks (this line ensures the joint connections remain)
                mp_drawing.draw_landmarks(
                    output_frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec
                )
                
                # Display overall status message
                if total_joints > 0 and reference_data:
                    # Calculate how many joints are incorrect
                    incorrect_joints = total_joints - correct_joints
                    
                    # Print detailed joint matching information
                    print(f"Correct Joints: {correct_joints} / {total_joints}")
                    
                    # Consider it a correct pose if 2 or fewer joints are incorrect
                    if incorrect_joints == 0:
                        status_msg = "PERFECT! GOOD JOB!"
                        status_color = (0, 255, 0)  # Green
                        # Add extra visual feedback for perfect pose
                        cv2.rectangle(output_frame, (frame.shape[1]//2 - 200, 10), 
                                     (frame.shape[1]//2 + 200, 50), (0, 255, 0), -1)
                        cv2.putText(
                            output_frame, status_msg, (frame.shape[1]//2 - 150, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
                        )
                    else:
                        # More than 2 joints are incorrect
                        status_msg = "Adjust your posture to match"
                        status_color = (0, 0, 255)  # Red
                        
                        cv2.putText(
                            output_frame, status_msg, (frame.shape[1]//2 - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA
                        )

            _, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route to the main index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for physio page
@app.route('/physio')
def physio():
    return render_template('physio.html')

@app.route('/general')
def general():
    return render_template('general-exercise.html')

# Route for belly selection page
@app.route('/belly')
def belly():
    return render_template('belly.html')

# Route for belly_perform (Final exercise page with camera feed)
@app.route('/belly_perform')
def belly_perform():
    return render_template('belly_perform.html')

@app.route('/balance')
def balance():
    return render_template('balance.html')

@app.route('/balance_perform')
def balance_perform():
    return render_template('balance-perform.html')

@app.route('/ball')
def ball():
    return render_template('ball.html')

@app.route('/ball_perform')
def ball_perform():
    return render_template('ball-perform.html')

@app.route('/resistance')
def resistance():
    return render_template('resistance.html')

@app.route('/resistance_perform')
def resistance_perform():
    return render_template('resistance-perform.html')

@app.route('/pose1')
def pose1():
    return render_template('pose1-cam.html')

@app.route('/pose2')
def pose2():
    return render_template('pose2-cam.html')

@app.route('/pose3')
def pose3():
    return render_template('pose3-cam.html')

@app.route('/pose4')
def pose4():
    return render_template('pose4-cam.html')

# Route for general video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routes for specific pose video feeds
@app.route('/video_feed/pose1')
def video_feed_pose1():
    return Response(generate_frames('pose1'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/pose2')
def video_feed_pose2():
    return Response(generate_frames('pose2'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/pose3')
def video_feed_pose3():
    return Response(generate_frames('pose3'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/pose4')
def video_feed_pose4():
    return Response(generate_frames('pose4'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/belly')
def video_feed_belly():
    return Response(generate_frames('belly'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/balance')
def video_feed_balance():
    return Response(generate_frames('balance'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/ball')
def video_feed_ball():
    return Response(generate_frames('ball'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/resistance')
def video_feed_resistance():
    return Response(generate_frames('resistance'), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)