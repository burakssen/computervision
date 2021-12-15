import cv2
import numpy as np
import moviepy.video.io.VideoFileClip as mpy
import imutils


vid = mpy.VideoFileClip('part1.1_out.mp4')
frame_count = vid.reader.nframes

video_fps = vid.fps


frames = []
for i in range(frame_count):
    frame = vid.get_frame(i*1.0/video_fps)
    frames.append(frame)


for i in range(len(frames) - 1):
    diff = cv2.absdiff(frames[i + 1], frames[i])
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Deneme", gray)
    cv2.waitKey(0)
