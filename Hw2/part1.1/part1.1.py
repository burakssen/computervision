import moviepy.video.io.VideoFileClip as mpy
from moviepy.editor import ImageSequenceClip
import numpy as np
import cv2

vid = mpy.VideoFileClip('shapes_video.mp4')
frame_count = vid.reader.nframes

video_fps = vid.fps


def median_filter(img, kernel_size):
    img_height = img.shape[0]
    img_width = img.shape[1]
    new_img = np.zeros((img_height, img_width), dtype=np.uint8)

    pad_size = int(kernel_size / 2)
    img_pad = np.pad(img, pad_size, 'edge')

    median = np.median(kernel_size)

    for i in range(img_height):
        for j in range(img_width):
            kernel = img_pad[i:i + kernel_size, j:j + kernel_size]
            median = np.median(kernel)
            new_img[i, j] = median

    return cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)


frames = []

for i in range(frame_count):
    print(i)
    frame = vid.get_frame(i*1.0/video_fps)
    frames.append(median_filter(frame, kernel_size=3).reshape((576, 720, 3)))

video = ImageSequenceClip(frames, fps=25)
video.write_videofile('part1_out.mp4', codec="libx264")
