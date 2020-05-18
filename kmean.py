import numpy as np
import cv2
from numpy import random
import copy
from scipy.spatial import distance


def getImage():
    img = cv2.imread('face.jpg')
    temp = np.array(img)
    return temp


def kmeans_algo(k=3, img=None):
    flag = 0
    kmean = [(0, 0, 0)]*k
    # initialising the random majors
    for i in range(k):
        l=np.random.randint(0, 220)
        m=np.random.randint(0, 200)
        kmean[i] = img[l][m]


    while flag != k:
        # this is used to store the point in specific mean category
        points = [[] for i in range(k)]
        pixels = [[] for i in range(k)]

        # calc. euclidean distance of each point and
        # place it accordingly in the points list of list
        for i in range(220):
            for j in range(200):
                temp = 0
                for l in range(k):
                    if distance.euclidean(img[i][j], kmean[l]) < distance.euclidean(img[i][j], kmean[temp]):
                        temp = l
                points[temp].append(img[i][j])
                pixels[temp].append((i, j))

        # avg of the k means
        for i in range(k):
            sum = (0, 0, 0)
            divisor = (len(points[i]), len(points[i]), len(points[i]))
            for j in range(len(points[i])):
                sum += points[i][j]
            avg = sum/divisor
            comparison = avg == kmean[i]
            if comparison.all():
                flag+=1
            kmean[i] = avg
        if flag == k:
            break
        else:
            flag = 0;
    return kmean, pixels

def saveImage(image, experiment, imgnum):
    cv2.imwrite("D:\\Downloads\\Sem 6\\AI\\Image-Compression-using-kmeans-" + str(experiment) + "_img" + str(imgnum) + ".png", image)
def __main__():
    temp = getImage()
    image = [[[0 for _ in range(3)] for _ in range(200)] for _ in range(220)]
    kmean, pixel = kmeans_algo(3, temp)

    for i in range(3):
        for j in pixel[i]:
            image[j[0]][j[1]]=kmean[i]

    saveImage(np.asarray(image),1, 1)

    return 0


__main__()
