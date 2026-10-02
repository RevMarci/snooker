import os
import cv2
from src.youtube_clip_downloader import YouTubeClipDownloader
from src.video_handler import VideoHandler

from src.mock_ball_detector import MockBallDetector

# Adjust this to change the video speed
_video_speed = 1.0
_display_scale = 0.5

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
        mock_detector = MockBallDetector()

        with VideoHandler(video_to_process) as video:
            print(f"Video paramaters: {video.width}x{video.height}, FPS: {video.fps}")
            
            for frame_index, frame in enumerate(video.frames()):
                
                # Example
                processed_frame = mock_detector.process_frame(frame)

                if _display_scale != 1.0:
                    new_width = int(video.width * _display_scale)
                    new_height = int(video.height * _display_scale)
                    display_frame = cv2.resize(processed_frame, (new_width, new_height))
                else:
                    display_frame = processed_frame
                
                cv2.imshow("Snooker Video", display_frame)
                
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
