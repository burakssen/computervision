# Name: Burak Şen
# ID: 150170063

import cv2
import scipy.io as io
import os
import numpy as np
from sklearn.metrics import precision_score, confusion_matrix
from operator import truediv


def load_mat(groundTruth):
    images = []
    for filename in os.listdir(groundTruth):
        ground = io.loadmat(os.path.join(groundTruth, filename))
        if ground is not None:
            images.append([ground, filename])
    return images


def load_images(image_path):
    images = []
    for filename in os.listdir(image_path):
        image = cv2.imread(os.path.join(
            f"{image_path + filename}"), cv2.IMREAD_GRAYSCALE)

        if image is not None:
            images.append([image, filename])
    return images


mats = load_mat("./data/groundTruth/")
imgs = load_images("./data/images/")

groundTruth = []
images = []


for mat in mats:
    boundaries = []

    for ground in mat[0]["groundTruth"]:
        for boundry in ground:
            bound = boundry[0][0]
            for matrix in bound:
                notBoundry = False

                if np.any(matrix > 1):
                    notBoundry = True

                if not notBoundry:
                    boundaries.append(matrix)
                    break

    final_boundary = boundaries[0]
    for i in range(1, len(boundaries)):
        final_boundary += boundaries[i]

    edges_255 = final_boundary * 255
    cv2.imwrite(f"./data/result/{mat[1]}.jpg", edges_255)
    edges_255[edges_255 >= 1] = 255
    groundTruth.append(edges_255)

for img in imgs:
    i = cv2.Canny(img[0], 225, 255)
    ret, i = cv2.threshold(
        i, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(f"./data/cannyResult/{img[1]}", i)
    images.append(i)

results = []

for i in range(len(images)):
    result = precision_score(
        groundTruth[i].flatten(), images[i].flatten(), average='binary', pos_label=255)
    results.append(result)

res = np.mean(results)
print("Average Precision:", res)
