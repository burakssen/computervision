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


firstFrame = None

for i in range(len(frames) - 1):
    frame_gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    frame_gray1 = cv2.GaussianBlur(frame_gray1, (1, 1), 0)
    frame_gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
    frame_gray2 = cv2.GaussianBlur(frame_gray2, (1, 1), 0)

    frame_diff = cv2.absdiff(frame_gray2, frame_gray1)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.dilate(thresh, None, iterations=2)
    cv2.imshow("Deneme", thresh)
    cv2.waitKey(0)
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    # cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours(cnts)
