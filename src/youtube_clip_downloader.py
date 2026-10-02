import os
import re
from urllib.parse import urlparse, parse_qs
import yt_dlp
from yt_dlp.utils import download_range_func


class YouTubeClipDownloader:
    def __init__(self):
        self.url = None
        self.start = None
        self.duration = None
        self.output = None

    @staticmethod
    def time_to_seconds(time_str):
        """Convert a time string to seconds. Supports formats like '1h2m3s', '1:02:03', '62', etc."""
        if isinstance(time_str, (int, float)):
            return int(time_str)

        time_str = str(time_str).strip()

        if any(c in time_str for c in ['h', 'm', 's']):
            hours = re.search(r'(\d+)h', time_str)
            minutes = re.search(r'(\d+)m', time_str)
            seconds = re.search(r'(\d+)s', time_str)
            
            h = int(hours.group(1)) if hours else 0
            m = int(minutes.group(1)) if minutes else 0
            s = int(seconds.group(1)) if seconds else 0
            return h * 3600 + m * 60 + s

        parts = list(map(int, time_str.split(":")))

        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds

        raise ValueError(f"Invalid time format: {time_str}")

    def read(self, provided_url=None):
        """Reads the user input for download parameters."""
        print("--- YouTube Clip Downloader ---")
        
        if provided_url:
            self.url = provided_url
            print(f"> Using provided URL: {self.url}")
        else:
            self.url = input("Please enter the YouTube URL: ").strip()

        # Process the URL to extract start time if present
        parsed_url = urlparse(self.url)
        query_params = parse_qs(parsed_url.query)

        if 't' in query_params:
            self.start = query_params['t'][0]
            print(f"> Detected start time from the link: {self.start} (automatically set).")
        else:
            self.start = input("Start time (e.g., 00:00:00 or 1:30): ").strip()

        # Duration
        duration_input = input("Duration (e.g., 1:00 or 15): ").strip()
        if not duration_input:
            self.duration = 60
        else:
            self.duration = duration_input
        
        # File name
        output_input = input("Output file name (leave empty for automatic 'clip[X].mp4'): ").strip()
        base_dir = "clips"
        
        if not output_input:
            i = 1
            while os.path.exists(os.path.join(base_dir, f"clip{i}.mp4")):
                i += 1
            self.output = os.path.join(base_dir, f"clip{i}.mp4")
        else:
            if not output_input.lower().endswith(".mp4"):
                output_input += ".mp4"
            
            filename = os.path.basename(output_input)
            self.output = os.path.join(base_dir, filename)
            
        print(f"> The file will be saved to: {self.output}")

    def run(self):
        """Runs the download based on the specified parameters."""
        if not self.url or not self.start or not self.duration:
            print("Error: Required information is missing. Please run the read() method first!")
            return

        print(f"\nPreparing download: {self.output}...")
        
        # Calculate time
        start_sec = self.time_to_seconds(self.start)
        duration_sec = self.time_to_seconds(self.duration)
        end_sec = start_sec + duration_sec

        output_dir = os.path.dirname(self.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # yt_dlp settings
        options = {
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "download_ranges": download_range_func(
                None,
                [(start_sec, end_sec)]
            ),
            "force_keyframes_at_cuts": True,
            "outtmpl": self.output,
            "quiet": False,
        }

        # Download
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([self.url])
            print("\nDownload completed successfully!")
        except Exception as e:
            print(f"\nAn error occurred during the download: {e}")
