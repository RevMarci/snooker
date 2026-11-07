import cv2
import random
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.table_top_view import TableTopView

# Add the video paths you want to save frames from
VIDEO_PATHS = [
    "source/clips/b.mp4",
    "source/clips/c.mp4",
]
# Number of frames to extract from each video
NUM_FRAMES_TO_EXTRACT = 10
#Input, output
OUTPUT_DIR = "source/random_frames"
WINDOW_NAME = "Dataset Frame Extractor"

# Display Configuration
DISPLAY_HEIGHT = 480  # Set the desired window height in pixels

user_action = None

def mouse_callback(event, x, y, flags, param):
    """Handles mouse click events to simulate UI buttons."""
    global user_action
    frame_height, button_width = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= frame_height:
            if x < button_width:
                user_action = 'save'
            else:
                user_action = 'discard'

def create_ui_image(frame):
    """Creates a composite image with the frame and a button panel at the bottom."""
    h, w, c = frame.shape
    panel_height = 80
    
    panel = np.zeros((panel_height, w, c), dtype=np.uint8)
    button_width = w // 2
    
    cv2.rectangle(panel, (0, 0), (button_width, panel_height), (255,255,255), -1)
    cv2.putText(panel, "save(Y)", (int(button_width * 0.1), int(panel_height * 0.6)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    cv2.rectangle(panel, (button_width, 0), (w, panel_height), (255,255,255), -1)
    cv2.putText(panel, "discard(N)", (button_width + int(button_width * 0.1), int(panel_height * 0.6)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    ui_img = np.vstack((frame, panel))
    return ui_img, panel_height, button_width

def get_user_decision(ui_img, frame_height, button_width):
    """Displays the UI and waits for keyboard or mouse input."""
    global user_action
    user_action = None
    
    cv2.imshow(WINDOW_NAME, ui_img)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=(frame_height, button_width))
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('y'):
            return 'save'
        elif key == ord('n'):
            return 'discard'
        elif key == 27:
            return 'quit'
        
        if user_action is not None:
            action = user_action
            user_action = None
            return action

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    table_top_processor = TableTopView()
    
    cv2.namedWindow(WINDOW_NAME)

    for video_path in VIDEO_PATHS:
        print(f"\n--- Starting to process video: {video_path} ---")
        
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            continue
            
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"Error: Video contains no frames: {video_path}")
            cap.release()
            continue

        frame_indices = list(range(total_frames))
        random.shuffle(frame_indices)

        saved_count = 0
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue

            processed_frame = table_top_processor.process(frame)
            
            if processed_frame is None:
                continue

            orig_h, orig_w = processed_frame.shape[:2]
            
            scale_ratio = DISPLAY_HEIGHT / float(orig_h)
            display_w = int(orig_w * scale_ratio)
            
            display_frame = cv2.resize(processed_frame, (display_w, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)

            ui_img, panel_height, button_width = create_ui_image(display_frame)
            frame_height = display_frame.shape[0]
            
            print(f"Showing frame {idx}. Progress for {video_filename}: {saved_count}/{NUM_FRAMES_TO_EXTRACT} saved.")
            
            decision = get_user_decision(ui_img, frame_height, button_width)
            
            if decision == 'save':
                output_name = f"{video_filename}_frame_{idx}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                
                cv2.imwrite(output_path, processed_frame)
                print(f"Saved: {output_path} (Original Resolution: {orig_w}x{orig_h})")
                
                saved_count += 1
                
                if saved_count >= NUM_FRAMES_TO_EXTRACT:
                    print(f"Reached target of {NUM_FRAMES_TO_EXTRACT} frames for {video_filename}.")
                    break
                    
            elif decision == 'quit':
                print("Exiting program requested by user.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

        cap.release()
        print(f"Finished processing video: {video_path}")

    cv2.destroyAllWindows()
    print("\nAll videos processed successfully.")

if __name__ == "__main__":
    main()
    