# -*- coding: utf-8 -*-
"""Video Segmentation and Tracking.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EC60joIneiwkxLEFvV_kN-C0Ttjpwhdb

# **Step 1: Data Collection**
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import shutil  # Import shutil module

# Check uploaded files
os.listdir()

import os

video_path = "/content/drive/My Drive/car video.mp4"

if os.path.exists(video_path):
    print("✅ File found:", video_path)
else:
    print("❌ File not found! Check if it's in the correct folder.")

import cv2

video_path = "/content/drive/My Drive/video.mp4"
cap = cv2.VideoCapture(video_path)  # Open the video

if cap.isOpened():
    print("✅ Video opened successfully!")
else:
    print("❌ Error opening video file!")

import cv2
import time
from google.colab.patches import cv2_imshow

video_path = "/content/drive/My Drive/car video.mp4"  # Ensure correct path

cap = cv2.VideoCapture(video_path)

start_time = time.time()  # Get start time

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or (time.time() - start_time) > 5:  # Stop after 5 seconds
        break

    cv2_imshow(frame)  # Display color frame

cap.release

"""# **Step 2 - Preprocessing & Feature Extraction**"""

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Feature Extraction using ORB
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    frame_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, color=(0,255,0))

    # Display results
    print(f"✅ Detected Keypoints: {len(keypoints)}")
    cv2_imshow(edges)  # Show edges
    cv2_imshow(frame_with_keypoints)  # Show keypoints

    break  # Process only the first frame

cap.release()
print("✅ Feature Extraction Completed!")

"""# Step 3: Motion-Based Segmentation
## (Using Optical Flow & Background Subtraction)

Optical Flow for Motion Segmentation
"""

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

cap = cv2.VideoCapture(video_path)

# Read first frame and convert to grayscale
ret, first_frame = cap.read()
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

    cv2_imshow(segmented_frame)  # Show motion segmentation

    prev_gray = gray.copy()  # Update previous frame

    break  # Process only first frame for now

cap.release()
print("✅ Motion Segmentation Completed!")

import cv2
import os

video_path = "/content/drive/My Drive/car video.mp4"  # ✅ Your correct path
output_dir = "/content/extracted_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_num = 0
saved = 0

if not cap.isOpened():
    print("❌ Failed to open video. Check path:", video_path)
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % 10 == 0:
            save_path = os.path.join(output_dir, f"frame_{frame_num}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"✅ Saved: {save_path}")
            saved += 1

        frame_num += 1

    cap.release()
    print(f"✅ Frame extraction completed! Total saved: {saved}")

print("Frame 0 exists:", os.path.exists("/content/extracted_frames/frame_0.jpg"))
print("Frame 10 exists:", os.path.exists("/content/extracted_frames/frame_10.jpg"))

import cv2
import os
import numpy as np
from google.colab.patches import cv2_imshow

# Set path to frames
frame_folder = "/content/extracted_frames"

# Load two consecutive frames
frame1 = cv2.imread(os.path.join(frame_folder, "frame_0.jpg"))
frame2 = cv2.imread(os.path.join(frame_folder, "frame_10.jpg"))

# Convert to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Calculate Dense Optical Flow (Farneback method)
flow = cv2.calcOpticalFlowFarneback(gray1, gray2,
                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Compute magnitude and angle of flow
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# Create HSV image (color-coded motion)
hsv = np.zeros_like(frame1)
hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
hsv[..., 1] = 255                      # Saturation
hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value

# Convert HSV to BGR for visualization
motion_mask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Display motion map
cv2_imshow(motion_mask)

"""# 4: Object Tracking using Optical Flow + Kalman Filter

Feature Tracking (Lucas-Kanade Optical Flow)
"""

import cv2
import numpy as np
import os

# Paths
frame_dir = "/content/extracted_frames"
output_dir = "/content/feature_tracking"
os.makedirs(output_dir, exist_ok=True)

# Params for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Params for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Load initial frames
frame0 = cv2.imread(os.path.join(frame_dir, "frame_0.jpg"))
gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

# Detect initial feature points
p0 = cv2.goodFeaturesToTrack(gray0, mask=None, **feature_params)

# Create mask image for drawing
mask = np.zeros_like(frame0)

for i in range(10, 101, 10):  # Track up to frame_100
    frame = cv2.imread(os.path.join(frame_dir, f"frame_{i}.jpg"))
    if frame is None:
        print(f"Frame {i} not found.")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray0, gray, p0, None, **lk_params)

    # Select good points
    if p1 is None:
        continue
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    cv2.imwrite(os.path.join(output_dir, f"tracked_{i}.jpg"), img)

    # Now update the previous frame and previous points
    gray0 = gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

print("✅ Feature tracking completed and saved in:", output_dir)

"""Kalman Filter for Object Tracking"""

import cv2
import numpy as np
import os

frame_dir = "/content/feature_tracking"
output_dir = "/content/kalman_tracking"
os.makedirs(output_dir, exist_ok=True)

# Kalman Filter initialization
kalman = cv2.KalmanFilter(4, 2)  # 4 variables: x, y, dx, dy; 2 measurements: x, y

# Transition matrix A
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)

kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
kalman.errorCovPost = np.eye(4, dtype=np.float32)

# Initialize state with first known point
init_frame = cv2.imread(os.path.join(frame_dir, "tracked_10.jpg"))
init_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
init_points = cv2.goodFeaturesToTrack(init_gray, maxCorners=1, qualityLevel=0.3, minDistance=7)
if init_points is not None:
    x, y = init_points[0, 0]
    kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)

# Start tracking
for i in range(10, 101, 10):
    frame_path = os.path.join(frame_dir, f"tracked_{i}.jpg")
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=1, qualityLevel=0.3, minDistance=7)

    if points is not None:
        meas = np.array([[np.float32(points[0][0][0])], [np.float32(points[0][0][1])]])
        kalman.correct(meas)

    pred = kalman.predict()
    pred_point = (int(pred[0]), int(pred[1]))

    # Draw predicted point
    cv2.circle(frame, pred_point, 10, (255, 0, 0), 2)  # Blue circle for Kalman
    cv2.putText(frame, f"Kalman: {pred_point}", (pred_point[0] + 10, pred_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(output_dir, f"kalman_{i}.jpg"), frame)

print("✅ Kalman tracking completed! Check folder:", output_dir)

"""Homography"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Input frames
frame1_path = "/content/extracted_frames/frame_0.jpg"
frame2_path = "/content/extracted_frames/frame_10.jpg"

img1 = cv2.imread(frame1_path)
img2 = cv2.imread(frame2_path)

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Use ORB for feature detection
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort by distance (good matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 50 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

# Get keypoints from best matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

# Compute Homography
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

# Warp image1 to align with image2
height, width, _ = img2.shape
warped = cv2.warpPerspective(img1, H, (width, height))

# Save results
os.makedirs("/content/homography", exist_ok=True)
cv2.imwrite("/content/homography/matched_features.jpg", img_matches)
cv2.imwrite("/content/homography/warped_frame.jpg", warped)

print("✅ Homography completed! Results saved in /content/homography")

import cv2
import os

frame_folder = "/content/extracted_frames"
output_video_path = "/content/reconstructed_video.avi"
fps = 10  # You can change this

# Get list of frames and sort
frames = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")],
                key=lambda x: int(x.split('_')[1].split('.')[0]))

# Load first frame to get dimensions
first_frame = cv2.imread(os.path.join(frame_folder, frames[0]))
height, width, _ = first_frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write frames
for frame_name in frames:
    frame_path = os.path.join(frame_folder, frame_name)
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()
print(f"✅ Video saved at: {output_video_path}")

import cv2
import numpy as np
import os

video_path = "/content/drive/My Drive/car video.mp4"  # or your correct path
cap = cv2.VideoCapture(video_path)

# Create folder to save
save_dir = "/content/report_images"
os.makedirs(save_dir, exist_ok=True)

# Process just the first frame
ret, frame = cap.read()
if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    frame_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, color=(255, 0, 0))

    # Save outputs
    cv2.imwrite(os.path.join(save_dir, "gray_frame.png"), gray)
    cv2.imwrite(os.path.join(save_dir, "canny_edges.png"), edges)
    cv2.imwrite(os.path.join(save_dir, "orb_keypoints.png"), frame_with_keypoints)

    print("✅ Images saved in:", save_dir)
else:
    print("❌ Could not read frame!")

cap.release()