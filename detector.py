import cv2 as cv
import numpy as np
import argparse
import logging

from utils import filterLabelNum, compileDataPoints, drawEllipse, computeCCL, displayImg
from ransac import RANSAC
from ellipse import Ellipse

if __name__ == "__main__" : 
    # Initializing parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--sample-size', type=int, default=5) 
    parser.add_argument('-k', '--iteration', type=int, default=50) 
    parser.add_argument('-t', '--threshold', type=float, default=0.1) 
    parser.add_argument('-d', '--tolerance', type=int, default=90) 
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args() 
    # Verbose setting 
    if args.verbose: 
        logging.basicConfig(level=logging.DEBUG, filename='', filemode='w')
    else : 
        logging.basicConfig(level=logging.INFO)
    # Reading image
    img = cv.imread('data/02.jpg')
    # Downscaling image 
    img = cv.resize(img, (640, 320))
    # Create a copy of input as an output
    out = img.copy()
    # Converting into edge map using Canny edge detection 
    edges = cv.Canny(out, 400, 600)
    # Connected Components Labelling 
    labelNum, label, stat, centroid = cv.connectedComponentsWithStats(edges, 8)
    # Discard small connected components 
    filteredLabelNum = filterLabelNum(stat, labelNum, 100)
    # Compute CCL image
    ccl = computeCCL(filteredLabelNum, label)
    ################## RANSAC PROGRAM ##################
    i = 1 # Counter 
    # Run RANSAC on each connected component
    for labelNum in filteredLabelNum : 
        # Compiling the coordinates of the pixels in the contour into data points
        dataPoints = compileDataPoints(label, labelNum)
        # Initialize an ellispe class 
        ellipse = Ellipse()
        # Run RANSAC on the data points 
        model = RANSAC(
            ellipse, 
            dataPoints, 
            args.sample_size, 
            args.iteration, 
            args.threshold, 
            args.tolerance
        )
        # Draw ellipse 
        if (model != []) :
            drawEllipse(out, model)

        i = i + 1
    ####################################################
    displayImg(img, edges, ccl, out)