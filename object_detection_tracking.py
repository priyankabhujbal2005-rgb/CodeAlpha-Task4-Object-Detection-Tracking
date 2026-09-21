
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# Load the pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Create Deep SORT tracker
tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_iou_distance=0.7
)

# Input video
input_video = "test_video.mp4"

# Output video
output_video = "final_tracked_video.mp4"

# Open video
cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

print("Processing video...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO object detection
    results = model(frame, verbose=False)

    detections = []

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            box_width = x2 - x1
            box_height = y2 - y1

            detections.append(
                (
                    [x1, y1, box_width, box_height],
                    confidence,
                    class_name
                )
            )

    # Update tracker
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    # Draw tracking results
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        class_name = track.get_det_class()

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display label and tracking ID
        label = f"{class_name} | ID: {track_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Save frame
    out.write(frame)

# Release resources
cap.release()
out.release()

print("Tracking completed!")
print("Output saved as:", output_video)
