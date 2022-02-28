## please add input images to this directory

import os
import cv2
import numpy as np
import moviepy.editor as mpy

images = []
gray_images = []
filenames = []


def read_images():
    for filename in os.listdir(os.getcwd() + "/images/"):
        filenames.append(filename)

    filenames.sort()

    for filename in filenames:
        image = cv2.imread(f"images/{filename}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        gray_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def ix(img1, img2, x, y):
    return (
        (int(img1[x + 1][y]) + int(img2[x + 1][y]) +
         int(img1[x + 1][y + 1]) + int(img2[x + 1][y + 1])) -
        (int(img1[x][y]) + int(img2[x][y]) +
         int(img1[x][y + 1]) + int(img2[x][y + 1]))
    ) / 4


def iy(img1, img2, x, y):
    return (
        (int(img1[x][y + 1]) + int(img2[x][y + 1]) +
         int(img1[x + 1][y + 1]) + int(img2[x + 1][y + 1])) -
        (int(img1[x][y]) + int(img2[x][y]) +
         int(img1[x + 1][y]) + int(img2[x + 1][y]))
    ) / 4


def it(img1, img2, x, y):
    return (
        (int(img1[x][y]) + int(img1[x][y + 1]) +
         int(img1[x + 1][y]) + int(img1[x + 1][y + 1])) -
        (int(img2[x][y]) + int(img2[x][y + 1]) +
         int(img2[x + 1][y]) + int(img2[x + 1][y + 1]))
    ) / 4


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def calc(img1, img2):
    A = []
    b = []
    for i in range(len(img1) - 1):
        for j in range(len(img1[0]) - 1):
            A.append([ix(img1, img2, i, j), iy(img1, img2, i, j)])
            b.append(it(img1, img2, i, j))

    return A, b


if __name__ == '__main__':
    read_images()

    row, col = gray_images[0].shape

    for k in range(len(gray_images) - 1):
        A = []
        b = []

        corners = cv2.goodFeaturesToTrack(
            gray_images[k], 0, 0.01, 10)

        corners = np.int0(corners)

        for corner in corners:

            x, y = corner.ravel()

            if x > 1 and y > 1:

                A, b = calc(gray_images[k][y - 2:y + 2, x - 2: x + 2],
                            gray_images[k + 1][y - 2:y + 2, x - 2: x + 2])

                A = np.asarray(A)
                b = -np.asarray(b)

                if is_invertible(np.dot(np.transpose(A), A)):

                    vec = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)),
                                 np.dot(np.transpose(A), b))

                    if np.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) >= 0.5:

                        cv2.arrowedLine(images[k], (x, y), (x - int(vec[1] * 40), y - int(vec[0] * 40)),
                                        color=(0, 0, 255), thickness=2, tipLength=0.2)

        cv2.imwrite(f"./output/{filenames[k]}", images[k])

    clip = mpy.ImageSequenceClip(images, fps=24)
    clip.write_videofile("movie.mp4", codec="png")
