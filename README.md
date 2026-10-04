# snooker

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

FFmpeg must be installed and added to your system PATH.
On Windows, you can install it using:

```bash
winget install Gyan.FFmpeg
```

You can check if it was installed correctly by running:

```bash
ffmpeg -version
```

Then run the program:

```bash
python main.py
```

Follow the instructions in the terminal.

You can use either a YouTube link or a filename if you have already downloaded the video.
For a YouTube link, you can use either a simple link or a link with a timestamp.  

Note: Recently ended YouTube livestreams may not work. If this happens, use an older video instead.

Recommended source for youtube videos:  
https://www.youtube.com/@snookerdresden2  
https://www.youtube.com/@snookerdresden3

There are some variables you can adjust at the top of main.py:

```python
_video_speed = 1.0
_display_scale = 0.5
```