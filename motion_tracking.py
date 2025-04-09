import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from google.colab.patches import cv2_imshow  # For Google Colab visualization

# Function to process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    processed_frames = []

    if not cap.isOpened():
        st.error("Error opening video file!")
        return None

    # Read the first frame and convert to grayscale
    ret, first_frame = cap.read()
    if not ret:
        st.error("Error reading the first frame!")
        cap.release()
        return None
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute Optical Flow (Farneback method)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert flow to magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold motion (highlight strong motion in green)
        motion_mask = np.zeros_like(frame)
        motion_mask[mag > 2] = [0, 255, 0]  # Highlight motion in green

        # Overlay motion mask on original frame
        segmented_frame = cv2.addWeighted(frame, 0.7, motion_mask, 0.3, 0)

        # Store processed frame
        processed_frames.append(segmented_frame)

        prev_gray = gray.copy()  # Update previous frame

    cap.release()
    return processed_frames

# Streamlit UI
st.title("Video Processing App")

# File uploader for video input
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name

    st.video(uploaded_video)  # Display the uploaded video

    st.markdown("### Processing video...")

    # Process the video and get frames
    processed_frames = process_video(temp_video_path)

    if processed_frames is not None:
        # Display processed frames in the app
        for frame in processed_frames:
            st.image(frame, channels="BGR", use_column_width=True)

        # Save the processed frames into a video
        output_video_path = "/content/processed_output_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 10, (frame.shape[1], frame.shape[0]))

        for frame in processed_frames:
            out.write(frame)

        out.release()

        # Provide download link for processed video
        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.avi",
                mime="video/avi"
            )
