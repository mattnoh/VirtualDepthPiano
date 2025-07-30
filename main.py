# sudo fuser -k /dev/video10
# sudo modprobe -r v4l2loopback
# sudo modprobe v4l2loopback exclusive_caps=1 video_nr=10 card_label="scrcpy" max_buffers=2
# Terminal 1: scrcpy --v4l2-sink=/dev/video10 --no-video-playback
# Terminal 2: ffplay -i /dev/video10

import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import math
from virtual_keyboard import VirtualKeyboard  
from keystate_tracker import KeyStateTracker

# --- Parameters ---
NUM_KEYS = 12
PRESS_THRESHOLD_DELTA = None  # Set by user input
RELEASE_THRESHOLD_DELTA = None  # Set by user input
MIN_PRESS_TIME = 0.1
KEYBOARD_WIDTH = 900
KEYBOARD_HEIGHT = 120
KEY_SPACING = 10

# --- MiDaS setup ---
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type != "MiDaS_small" else midas_transforms.small_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# --- MediaPipe Hands setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- Helper functions ---
def get_midas_depth(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()
    return prediction

def estimate_desk_depth(depth_map, desk_region=(slice(-50, -10), slice(None))):
    # Median depth in bottom strip of image (this represents the desk/table surface)
    region = depth_map[desk_region]
    return np.median(region)

def is_fingertip_touching_desk(fingertip_depth, desk_depth, press_threshold=PRESS_THRESHOLD_DELTA):
    """
    Check if fingertip is at desk level (touching the surface)
    In MiDaS depth maps, smaller values = closer to camera, larger values = further away
    The desk surface should have a consistent depth value
    Fingertips touching the desk should have depth values very close to the desk depth
    """
    # Fingertip must be at or slightly beyond the desk level
    # Allow small threshold for noise and slight pressure into the surface
    return abs(fingertip_depth - desk_depth) <= press_threshold

def smooth_depth(current_depth, previous_depth, alpha=0.7):
    """Smooth depth values to reduce noise"""
    if previous_depth is None:
        return current_depth
    return alpha * current_depth + (1 - alpha) * previous_depth

# --- Main loop ---
cap = cv2.VideoCapture("/dev/video10")

# Initialize variables
desk_depth = None
key_tracker = KeyStateTracker(NUM_KEYS)
finger_depths = {}
virtual_keyboard = None

# Calibration phase
print("Calibrating desk depth... Keep hands away from desk for 3 seconds")
calibration_frames = []
calibration_start = time.time()

# Threshold input status
thresholds_set = False
input_prompt_displayed = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    
    # Initialize virtual keyboard at center of screen
    if virtual_keyboard is None:
        virtual_keyboard = VirtualKeyboard(w // 2, h // 2, KEYBOARD_WIDTH, KEYBOARD_HEIGHT, NUM_KEYS, KEY_SPACING)
    
    depth_map = get_midas_depth(frame)
    current_time = time.time()
    
    # Calibration phase
    if desk_depth is None:
        if current_time - calibration_start < 3.0:
            calibration_frames.append(depth_map)
            cv2.putText(frame, f"Calibrating... {3.0 - (current_time - calibration_start):.1f}s", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            avg_depth_map = np.mean(calibration_frames, axis=0)
            desk_depth = estimate_desk_depth(avg_depth_map)
            print(f"Desk depth calibrated: {desk_depth:.3f}")

    # Get thresholds from user after calibration
    if desk_depth is not None and not thresholds_set:
        if not input_prompt_displayed:
            print("\nCalibration complete. Please enter thresholds in the terminal.")
            input_prompt_displayed = True
            
        # Show message on screen
        cv2.putText(frame, "Calibration complete.", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Switch to terminal to input thresholds", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Get thresholds from terminal
        if PRESS_THRESHOLD_DELTA is None:
            try:
                PRESS_THRESHOLD_DELTA = float(input("Enter press threshold delta (e.g., 6.0): "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
        
        if RELEASE_THRESHOLD_DELTA is None:
            try:
                RELEASE_THRESHOLD_DELTA = float(input("Enter release threshold delta (e.g., 7.5): "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
        
        thresholds_set = True
        print(f"Thresholds set - Press: {PRESS_THRESHOLD_DELTA}, Release: {RELEASE_THRESHOLD_DELTA}")
        print("Starting detection...")

    # Main detection loop
    if thresholds_set:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                fingertip_indices = [4, 8, 12, 16, 20]  # thumb to pinky
                
                for finger_idx, fingertip_idx in enumerate(fingertip_indices):
                    fingertip = hand_landmarks.landmark[fingertip_idx]
                    x_px = int(fingertip.x * w)
                    y_px = int(fingertip.y * h)

                    # Check boundary
                    if 0 <= x_px < w and 0 <= y_px < h:
                        raw_depth = depth_map[y_px, x_px]
                        
                        # Smooth the depth reading
                        finger_key = (hand_idx, finger_idx)
                        smoothed_depth = smooth_depth(raw_depth, finger_depths.get(finger_key))
                        finger_depths[finger_key] = smoothed_depth
                        
                        # Get key index from virtual keyboard (None if outside bounds)
                        key_idx = virtual_keyboard.get_key_from_point(x_px, y_px)
                        
                        # Check if fingertip is actually touching the desk surface
                        is_touching_desk = is_fingertip_touching_desk(smoothed_depth, desk_depth, PRESS_THRESHOLD_DELTA)
                        
                        # Update finger state and check for new press
                        unique_finger_id = hand_idx * 5 + finger_idx
                        new_press = key_tracker.update_finger_state(
                            unique_finger_id, key_idx, is_touching_desk, current_time
                        )
                        
                        # Visual feedback for fingertips
                        if virtual_keyboard.is_point_in_keyboard(x_px, y_px):
                            if is_touching_desk:
                                # Red circle for touching desk within keyboard
                                cv2.circle(frame, (x_px, y_px), 12, (0, 0, 255), -1)
                                # Show depth difference from desk
                                depth_diff = abs(smoothed_depth - desk_depth)
                                cv2.putText(frame, f"TOUCH {depth_diff:.3f}", (x_px + 15, y_px - 15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            else:
                                # Blue circle for hovering over keyboard (not touching)
                                cv2.circle(frame, (x_px, y_px), 10, (255, 0, 0), -1)
                                # Show how far from desk surface
                                depth_diff = abs(smoothed_depth - desk_depth)
                                cv2.putText(frame, f"HOVER {depth_diff:.3f}", (x_px + 15, y_px - 15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        else:
                            # Gray circle for outside keyboard area
                            cv2.circle(frame, (x_px, y_px), 6, (128, 128, 128), -1)
                            # Show depth info
                            depth_diff = abs(smoothed_depth - desk_depth)
                            cv2.putText(frame, f"{depth_diff:.3f}", (x_px + 10, y_px - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
                        
                        # Print new key press
                        if new_press:
                            print(f"Key {key_idx + 1} pressed by finger {finger_idx + 1} (Hand {hand_idx + 1})")

        # Draw the virtual keyboard with active keys highlighted
        active_keys = key_tracker.get_active_keys()
        virtual_keyboard.draw(frame, active_keys)

        # Show status information with desk depth reference
        cv2.putText(frame, f"Desk Level: {desk_depth:.3f} | Press Δ: {PRESS_THRESHOLD_DELTA} | Release Δ: {RELEASE_THRESHOLD_DELTA}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, "RED = Touching Desk | BLUE = Hovering | GRAY = Outside Keyboard", 
                   (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show active key numbers
        if active_keys:
            active_text = "Active: " + ", ".join([str(k+1) for k in sorted(active_keys)])
            cv2.putText(frame, active_text, (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Virtual AR Piano Keyboard", frame)
    
    # Show depth heatmap
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    cv2.imshow("Depth Heatmap", depth_colormap)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()