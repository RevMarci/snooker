from src.YouTubeClipDownloader import YouTubeClipDownloader

def main():
    print("Run snooker detection...\n")
    
    downloader = YouTubeClipDownloader()
    
    # Get video data from user
    downloader.read()
    
    downloader.run()
    
    print("\nProgram completed.")

if __name__ == "__main__":
    main()