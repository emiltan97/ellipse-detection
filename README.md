# Ellipse Detection using RANSAC 
## Overview 
A program that detects elliptical shape in an image. The system pipeline : 
- Downscaling an image for shorter processing time
- Convert the input image into an edge map using Canny edge detection
- Two pass connected components labelling to remove small blobs and noises
- Run ellipse fitting with RANSAC on the image 
## Running the program 
### To process a single image
```console
python detector.py --<filename> --display
```
### To process a folder of images 
```console
python detector.py --<dirname> -- display 
```
### Input format
- Single JPG image.
- A folder of JPG images.
## Sample 
**Input** : 
![](https://github.com/emiltan97/ellipse-detection/blob/master/data/02.JPG)
![](https://github.com/emiltan97/ellipse-detection/blob/master/data/03.JPG) 
**Output** :
![](https://github.com/emiltan97/ellipse-detection/blob/master/out/02.png)
![](https://github.com/emiltan97/ellipse-detection/blob/master/out/03.png)
## Requirement 
- Python 3.8 
- OpenCV 
- matplotlib 
