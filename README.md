# EaseFit Pose Estimation Flask Application

## Overview
This project is a Flask-based web application that utilizes OpenCV and MediaPipe Pose to detect human poses from live video streams. The application extracts joint angles and compares them with predefined reference poses to provide feedback on pose accuracy.

## Features
- **Real-time Pose Detection**: Uses OpenCV and MediaPipe Pose to detect human poses.
- **Angle Calculation**: Computes angles between key joints such as shoulders, elbows, hips, knees, and wrists.
- **Reference Pose Comparison**: Compares detected poses with predefined reference poses.
- **Web Interface**: Streams video and displays pose analysis using Flask.

## Installation
### Prerequisites
Ensure you have Python 3.7+ installed.

### Install Dependencies
Run the following command to install the required packages:
```bash
pip install flask opencv-python mediapipe numpy
```

## Usage
1. **Start the Flask Server**:
   ```bash
   python app.py
   ```
2. **Access the Web Interface**:
   Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```
3. **Pose Detection**:
   - The webcam feed will display the detected pose.
   - Angles between joints will be calculated and compared to reference poses.

## Project Structure
```
/pose_estimation_project
│── app.py                  # Main Flask application
│── static/
│   ├── reference_poses/    # Folder containing reference pose images
│   ├── annotated_reference_pose/  # Folder for storing annotated pose images
│── templates/
│   ├── index.html          # Web interface template
│── requirements.txt        # List of dependencies
│── README.md               # Documentation
```

## Key Functions
### `calculate_angle(a, b, c)`
- Calculates the angle between three points using the arctangent function.

### `extract_joint_angles(landmarks)`
- Extracts angles for key joints from detected pose landmarks.

### `extract_reference_angles(reference_image_path, pose_type)`
- Extracts joint angles from reference images for pose comparison.

### `get_reference_angles(pose_type)`
- Retrieves reference angles for a specific pose from cached values or processes a reference image.

## Reference Pose Folder Structure
- Store images in `static/reference_poses/` with filenames matching pose types (e.g., `pose1.jpg`).

## Troubleshooting
- If pose detection is not working, ensure your webcam is accessible.
- Check if `static/reference_poses/` contains the required images.
- Ensure dependencies are correctly installed.

## Future Improvements
- Implement feedback for incorrect poses.
- Enhance UI for better user experience.
- Add support for more complex pose detection scenarios.

## Disclaimer
This project is a prototype developed by my teammates and me for our first hackathon. As this is an early-stage project, please note that it may have limitations, and we are still learning and improving our skills in this field. Thank you for your understanding!

## License
This project is open-source and available for modification and distribution.

