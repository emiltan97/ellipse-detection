import numpy as np
import cv2 as cv

from math import sin, cos
from matplotlib import pyplot as plt

def getDistance(data, x0, y0, a, b, theta) : 

    x = data[0] 
    y = data[1]

    mat1 = np.array([
        [x - x0, y - y0]
    ])
    mat2 = np.array([
        [ cos(theta), sin(theta)],
        [-sin(theta), cos(theta)]  
    ])
    mat3 = np.array([
        [1 / a**2, 0],
        [0, 1 / b**2]
    ])
    mat4 = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta),  cos(theta)]
    ])
    mat5 = np.array([
        [x - x0],
        [y - y0]
    ])

    dist = abs(
        (mat1 @ mat2 @ mat3 @ mat4 @ mat5) - 1
    )    
    
    return dist[0][0]

def compileDataPoints(label, labelNum) : 

    dataPoints = []

    for i in range(0, len(label)) :
        for j in range(0, len(label[i])) :
            if label[i][j] == labelNum :
                dataPoints.append((j, i))

    dataPoints = np.array(dataPoints)

    return dataPoints

def filterLabelNum(stat, labelNum, minSize) :

    filteredLabelNum = []

    sizes = stat[1:, -1]

    for i in range(1, labelNum) :
        if sizes[i-1] >= minSize : 
            filteredLabelNum.append(i)

    return filteredLabelNum

def drawEllipse(img, model, isCrop) : 

    x0 = int(model[0])
    y0 = int(model[1])
    a = int(model[2])
    b = int(model[3])
    theta = int(model[4])

    if isCrop : 
        cv.ellipse(
            img=img, 
            center=(x0, y0), 
            axes=(a, b), 
            angle=theta, 
            startAngle=0, 
            endAngle=360, 
            color=(255, 255, 255), 
            thickness=-1
        )
    else : 
        cv.ellipse(
            img=img, 
            center=(x0, y0), 
            axes=(a, b), 
            angle=theta, 
            startAngle=0, 
            endAngle=360, 
            color=(0, 0, 255), 
            thickness=2
        )

def computeCCL(labelNum, label):
    for i in range(0, len(label)) :
        for j in range(0, len(label[i])) :
            if label[i][j] not in labelNum :
                label[i][j] = 0

    labelCol = np.uint8(255 * label/np.max(label))
    blankCol = 255 * np.ones_like(labelCol)
    ccl      = cv.merge([labelCol, blankCol, blankCol])
    ccl[labelCol == 0] = 0 

    return ccl

def displayImg(img, edges, ccl, out) : 
    plt.subplot(221)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Input')
    plt.subplot(222)
    plt.imshow(cv.cvtColor(edges, cv.COLOR_BGR2RGB))
    plt.title('Edge Map')
    plt.subplot(223)
    plt.imshow(ccl)
    plt.title('Connected Component Labelling (CCL)')
    plt.subplot(224)
    plt.imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
    plt.title('Output')

    plt.show()

def makeTransparent(img) : 

    img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

    for i in range(0, img.shape[0]) :
        for j in range(0, img.shape[1]) :
            pixel = img[i][j]
            if not pixel[:3].any() : 
                pixel[3] = 0
                
    return img