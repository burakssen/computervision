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
        ret, image = cv2.threshold(
            image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if image is not None:
            images.append([image, filename])
    return images


mats = load_mat("./data/groundTruth/")
imgs = load_images("./data/images/")

groundTruth = []

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
    ret, edges_255 = cv2.threshold(
        edges_255, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    groundTruth.append(edges_255)

results = []

for i in range(len(imgs)):
    result = precision_score(groundTruth[i].flatten(), imgs[i][0].flatten(
    ), average='binary', pos_label=255)
    results.append(result)

res = np.mean(results)
print("Average Precision:", res)
