
# Object Detection and Tracking using YOLO and Deep SORT

## Project Overview

This project implements object detection and tracking using a pre-trained YOLO model and Deep SORT.

The system processes a video frame by frame, detects objects, and assigns unique tracking IDs to the detected objects.

This project was developed as part of the CodeAlpha Artificial Intelligence Internship.

## Objectives

- Detect objects in a video using YOLO.
- Draw bounding boxes around detected objects.
- Track detected objects across video frames.
- Assign unique IDs to tracked objects.
- Generate a processed output video.

## Technologies Used

- Python
- YOLO
- OpenCV
- Deep SORT
- Google Colab

## Project Workflow

Input Video
    ↓
OpenCV
    ↓
YOLO Object Detection
    ↓
Bounding Boxes
    ↓
Deep SORT Tracking
    ↓
Tracking IDs
    ↓
Output Video

## How It Works

1. The input video is read using OpenCV.
2. YOLO detects objects in each video frame.
3. The detected objects are passed to the Deep SORT tracker.
4. Deep SORT tracks the detected objects between frames.
5. Each tracked object receives a unique ID.
6. Bounding boxes, object labels, and tracking IDs are displayed.
7. The processed video is saved as the final output.

## Project Files

- object_detection_tracking.py - Main Python program.
- requirements.txt - Required Python libraries.
- README.md - Project documentation.
- final_tracked_video.mp4 - Processed output video.

## Installation

Install the required libraries using:

pip install ultralytics opencv-python deep-sort-realtime

## Running the Project

Run the following command:

python object_detection_tracking.py

The program processes the input video and creates the tracked output video.

## Output

The final video contains:

- Object bounding boxes
- Object class labels
- Unique tracking IDs

Example:

person | ID: 1
car | ID: 2

## Future Improvements

- Real-time webcam detection
- Object counting
- Improved tracking accuracy
- Support for multiple video sources
- Performance optimization

## Project Information

Artificial Intelligence Internship Project

Internship: CodeAlpha
Task: Object Detection and Tracking
