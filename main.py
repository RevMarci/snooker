import cv2
from src.youtube_clip_downloader import YouTubeClipDownloader
from src.video_handler import VideoHandler

# Adjust this to change the video speed
_video_speed = 1.0 

def main():
    print("Run snooker detection...\n")
    
    downloader = YouTubeClipDownloader()
    
    # Get video data from user
    downloader.read()
    
    downloader.run()

    if not downloader.output:
        print("Error: No downloaded file found.")
        return
    
    # Process the video
    try:
        with VideoHandler(downloader.output) as video:
            print(f"Video paramaters: {video.width}x{video.height}, FPS: {video.fps}")
            
            for frame_index, frame in enumerate(video.frames()):
                
                # Example
                # detections = detector.process_frame(frame)
                
                # Test display
                cv2.imshow("Snooker Video", frame)
                
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