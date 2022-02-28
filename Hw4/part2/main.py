import os
import cv2
import numpy as np
import moviepy.editor as mpy

images = []
filenames = []


def read_images():
    for filename in os.listdir(os.getcwd() + "/images/"):
        filenames.append(filename)

    filenames.sort()

    for filename in filenames:
        image = cv2.imread(f"images/{filename}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


new_Images = []

if __name__ == "__main__":
    read_images()

    background = cv2.imread("images/00459.png", 0)
    # background = np.asarray(background)

    threshold = 60

    results = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        background = cv2.GaussianBlur(background, (5, 5), 0)

        result = cv2.subtract(background, gray_image)

        result[result >= threshold] = 255
        result[result < threshold] = 0
        kernel = np.asarray([[1, 2, 1], [2, 3, 2], [1, 2, 2]])

        result = cv2.dilate(result, kernel, iterations=1)

        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        results.append(result)

    clip = mpy.ImageSequenceClip(results, fps=24)
    clip.write_videofile("movie.mp4", codec="png")
