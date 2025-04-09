import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.title("Motion Detection and Highlighting")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    ret, initial_frame = cap.read()
    if not ret:
        st.error("Unable to read the initial frame.")
    else:
        previous_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        counter = 1
        total_saved = 0
        output_frames = []

        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            if counter % 2 == 0:
                gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                frame_difference = cv2.absdiff(previous_gray, gray_current)
                _, motion_binary = cv2.threshold(frame_difference, 25, 255, cv2.THRESH_BINARY)
                motion_colored = cv2.cvtColor(motion_binary, cv2.COLOR_GRAY2BGR)
                motion_colored[motion_binary == 255] = [0, 255, 0]
                blended_output = cv2.addWeighted(current_frame, 0.7, motion_colored, 0.3, 0)
                output_frames.append(cv2.cvtColor(blended_output, cv2.COLOR_BGR2RGB))
                previous_gray = gray_current.copy()
                total_saved += 1

            counter += 1

        st.success(f"Processed {total_saved} frames with motion detection.")
        for frame in output_frames[:10]:
            st.image(frame, use_column_width=True)

        save_video = st.button("Generate Final Output Video")

        if save_video:
            temp_video_path = os.path.join(tempfile.gettempdir(), "final_motion_output.mp4")
            height, width, _ = output_frames[0].shape
            out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
            for frame in output_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            with open(temp_video_path, "rb") as file:
                st.download_button("Download Final Video", file, file_name="motion_output.mp4")
