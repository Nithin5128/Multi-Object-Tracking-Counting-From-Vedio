# Import application framework and processing libraries
import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

# Configure Streamlit page settings
st.set_page_config(page_title="Multi-Object Tracking & Counting", layout="wide")
st.title("Multi-Object Tracking & Counting Dashboard")
st.markdown("Real-time detection via **YOLOv8**, spatial-temporal association via **ByteTrack**, and tripwire counting.")

# Sidebar configuration controls
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
line_pos_pct = st.sidebar.slider("Tripwire Vertical Position (%)", 20, 80, 60, 5)

# Video selection: File upload or local fallback
uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
default_video_path = "input_vedio.mp4"

video_source = None
if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    video_source = temp_file.name
elif os.path.exists(default_video_path):
    video_source = default_video_path

# Display metrics and video frame columns
metric_col, video_col = st.columns([1, 3])
with metric_col:
    count_metric = st.empty()
    count_metric.metric(label="Total Objects Counted", value=0)
with video_col:
    video_display = st.empty()

# Start inference button
if st.sidebar.button("Run Tracking & Counting"):
    if not video_source:
        st.error("Please upload a video or place input_vedio.mp4 in the folder.")
    else:
        # Load YOLOv8 Nano model
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(video_source)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = int(frame_height * (line_pos_pct / 100.0))

        track_history = {}
        counted_ids = set()
        total_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run detection + ByteTrack
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_threshold,
                classes=[0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
            )

            # Check for tracked boxes
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = box
                    label = model.names[cls_id]

                    # Ground contact point
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)

                    # Annotations
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} #{track_id}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                    # Downward crossing logic
                    if track_id in track_history:
                        prev_y = track_history[track_id]
                        if prev_y < line_y and cy >= line_y and track_id not in counted_ids:
                            counted_ids.add(track_id)
                            total_count += 1
                            count_metric.metric(label="Total Objects Counted", value=total_count)

                    track_history[track_id] = cy

            # Draw counting line
            cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 255), 2)

            # Render frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_display.image(rgb_frame, channels="RGB", use_container_width=True)

        cap.release()
        st.success(f"Processing Complete! Final Count: {total_count}")