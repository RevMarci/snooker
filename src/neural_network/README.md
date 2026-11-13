# How to train your neural network

## Setup

- Use the ```root/scripts/extract_frames.py``` script to extract frames from already downloaded videos. You can download videos with the main program. You can choose the videos and set the number of frames to download in the script.  
- Go to ```https://app.cvat.ai/tasks```, and log in. Create a ```Project``` and make one rectangle label called ```ball```. Create a new task, and upload the extracted frames. It'll create a job, open it.

## Labeling

- Use the ```Draw a rectangle``` tool with ```2 Points``` method. Draw a rectangle around each ball (not around the ball shadow, just the ball). You can move around with left click and zoom in/out with the scroll wheel. After you draw a rectangle, it will drop the tool, you can "grab" it again with ```N```. You can jump between frames with ```D``` and ```F```.
- After you done, export the dataset in ```Ultralytics YOLO Detection 1.0```.
- In the dataset you will have the following:
- train.txt: you won't need this
- data.yaml: change its content to the data.yaml next to this README
- In the ```labels/train``` folder you will find all the labels. Next to this folder make a ```test``` and a ```val``` folder.
- Distribute the downloaded labels between these folders with the following ratios: ```train - 70%, test - 15%, val - 15%```
- Next to the ```labels``` folder make an ```images``` folder, with the same ```train```, ```test```, ```val``` folders inside them.
- Copy the extracted images into the corresponding ```images/train```, ```images/test``` and ```images/val``` folders, following the same distribution as in the ```labels``` folder. Each image must be placed in the same folder as its corresponding label.

## Training

- You can train on your own computer or in Google Colab
- For the training follow this link: https://colab.research.google.com/drive/10EYPOhxNtPhrjQOXH5HzFn9Ck7dS9-kV?usp=sharing
- After running this, it will create files, and statistics in your Google Drive. Download the ```best.pt```. This is your trained neural network.