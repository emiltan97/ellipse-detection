import cv2 as cv
import numpy as np
import argparse
import logging
import os

from utils import filterLabelNum, compileDataPoints, drawEllipse, computeCCL, displayImg
from ransac import RANSAC
from ellipse import Ellipse
from datetime import datetime

if __name__ == "__main__" : 
    # Initializing parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--sample-size', type=int, default=5) 
    parser.add_argument('-k', '--iteration',   type=int, default=50) 
    parser.add_argument('-t', '--threshold',   type=float, default=0.1) 
    parser.add_argument('-d', '--tolerance',   type=int, default=90) 
    parser.add_argument('-w', '--width',       type=int, default=640)
    parser.add_argument('-hi', '--height',     type=int, default=320)
    parser.add_argument('-a', '--blob-area',   type=int, default=100)
    parser.add_argument('-v', '--verbose',     action='store_true')
    parser.add_argument('-t1', '--lower-threshold', type=int, default=400)
    parser.add_argument('-t2', '--upper-threshold', type=int, default=600)
    parser.add_argument('-fn', '--filename', type=str)
    parser.add_argument('--dirname', type=str, default='data/')
    parser.add_argument('--out-dir', type=str, default='out/')
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args() 
    # Verbose setting 
    if args.verbose: 
        LOG_FILENAME = datetime.now().strftime('log%H%M%S%d%m%Y.log')
        logging.basicConfig(level=logging.DEBUG, filename=LOG_FILENAME, filemode='w')
    else : 
        logging.basicConfig(level=logging.INFO)
    ################## PREPROCESSING ###################
    if args.filename :
        filename = args.filename
    for files in os.listdir(args.dirname) : 
        if files.endswith('.JPG') : 
            if not args.filename :
                filename = args.dirname + files 
            # Reading image
            img = cv.imread(filename)
            logging.info(f'File          : {filename}')
            # Downscaling image 
            img = cv.resize(img, (args.width, args.height))
            logging.info(f'Size (w, h)   : {args.width}, {args.height} ')
            # Create a copy of input as an output
            out = img.copy()
            # Converting into edge map using Canny edge detection 
            edges = cv.Canny(out, args.lower_threshold, args.upper_threshold)
            logging.info(f'Edge (t1, t2) : {args.lower_threshold}, {args.upper_threshold}')
            # Connected Components Labelling 
            labelNum, label, stat, centroid = cv.connectedComponentsWithStats(edges, 8)
            # Discard small connected components 
            filteredLabelNum = filterLabelNum(stat, labelNum, args.blob_area)
            logging.info(f'Blob Area     : {args.blob_area}')
            # Compute CCL image
            ccl = computeCCL(filteredLabelNum, label)
            ################## RANSAC PROGRAM ##################
            i = 1 # Counter 
            # Run RANSAC on each connected component
            for labelNum in filteredLabelNum : 
                logging.debug(f'COMPONENT : {i}')
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

                logging.info(f'Component {i} process complete.')
                i = i + 1
            ####################################################
            # Save images 
            if not os.path.isdir(args.out_dir) : 
                os.mkdir(args.out_dir)
            cv.imwrite(args.out_dir + filename.replace('.JPG', '.png'), out)
            # Display image
            if args.display :
                displayImg(img, edges, ccl, out)
            # End the loop if its a single file 
            if args.filename: 
                break