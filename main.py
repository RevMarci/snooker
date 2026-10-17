import os
import cv2
import numpy as np
from src.youtube_clip_downloader import YouTubeClipDownloader
from src.video_handler import VideoHandler
from src.perspective_corrector import PerspectiveCorrector

from src.color_based_ball_detector import ColorBasedBallDetector
from src.neural_network_ball_detector import NeuralNetworkBallDetector

# Adjust this to change the video speed
_video_speed = 1.0
_display_scale = 0.5

SHOW_ORIGINAL = False
SHOW_CORRECTED = False
SHOW_PROCESSED = True

ENABLE_COLOR_DETECTOR = True
ENABLE_NEURAL_DETECTOR = True

def main():
    print("Run snooker detection...\n")
    
    user_input = input("Enter video file path or YouTube URL (e.g., clip1.mp4 or https://youtu.be/...): ").strip()
    
    video_to_process = None

    if user_input.startswith(("http://", "https://", "www.", "youtu.be")):
        print("\n> YouTube link detected...")
        downloader = YouTubeClipDownloader()
        
        downloader.read(provided_url=user_input)
        downloader.run()

        if not downloader.output:
            print("Error: No downloaded file found.")
            return
            
        video_to_process = downloader.output
        
    else:
        print("\n> Local file name detected. Searching for the file...")
        if not user_input.lower().endswith(".mp4"):
            user_input += ".mp4"
            
        if os.path.exists(user_input):
            video_to_process = user_input
        elif os.path.exists(os.path.join("clips", user_input)):
            video_to_process = os.path.join("clips", user_input)
        else:
            print(f"Error: The file '{user_input}' was not found in the current or 'clips/' directory.")
            return

    print(f"\n> Video selected for processing: {video_to_process}\n")
    
    # Process the video
    try:
        color_based_detector = ColorBasedBallDetector()
        neural_network_detector = NeuralNetworkBallDetector()

        perspective_corrector = PerspectiveCorrector()

        with VideoHandler(video_to_process) as video:
            print(f"Video paramaters: {video.width}x{video.height}, FPS: {video.fps}")
            
            for frame_index, frame in enumerate(video.frames()):
                # Frame correction
                corrected_frame = perspective_corrector.process(frame)
                
                # == MAIN PROCESSING ==
                combined_overlay = np.zeros_like(corrected_frame)

                if ENABLE_COLOR_DETECTOR:
                    overlay_color = color_based_detector.process_frame(corrected_frame)
                    combined_overlay = cv2.add(combined_overlay, overlay_color)

                if ENABLE_NEURAL_DETECTOR:
                    overlay_neural = neural_network_detector.process_frame(corrected_frame)
                    combined_overlay = cv2.add(combined_overlay, overlay_neural)

                processed_frame = cv2.add(corrected_frame, combined_overlay)
                # == END OF MAIN PROCESSING ==

                if _display_scale != 1.0:
                    new_width = int(video.width * _display_scale)
                    new_height = int(video.height * _display_scale)
                else:
                    new_width, new_height = video.width, video.height

                # Display selected windows
                if SHOW_ORIGINAL:
                    disp_orig = cv2.resize(frame, (new_width, new_height)) if _display_scale != 1.0 else frame
                    cv2.imshow("Snooker Video (Original)", disp_orig)

                if SHOW_CORRECTED:
                    disp_corr = cv2.resize(corrected_frame, (new_width, new_height)) if _display_scale != 1.0 else corrected_frame
                    cv2.imshow("Snooker Video (Perspective Corrected)", disp_corr)

                if SHOW_PROCESSED:
                    disp_proc = cv2.resize(processed_frame, (new_width, new_height)) if _display_scale != 1.0 else processed_frame
                    cv2.imshow("Snooker Video (Processed)", disp_proc)
                
                # Escape key: Q
                if cv2.waitKey(int(1000 / video.fps / _video_speed)) & 0xFF == ord('q'):
                    print("Video playback interrupted by user.")
                    break
                    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
    
    print("\nProgram completed.")

if __name__ == "__main__":
    main()
