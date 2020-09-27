import numpy as np

from itertools import combinations
from utils import getDistance

def RANSAC(model, data, sampleSize, iterations, threshold, tolerance)  : 
    # data        -> data points 
    # sample_size -> n, smallest number of points required 
    # iterations  -> k, the number of iterations required 
    # threshold   -> t, threshold used to identify a point that fits well 
    # tolerance   -> d, number of nearby points required to assert a model fits well

    iteration = 0 
    bestModel = []
    bestError = np.Inf

    # Until k iterations have occurred
    while iteration < iterations : 
        print(f"Iteration : {iteration}")
        # Draw a sample of n points from the data uniformly and at random 
        print("Choosing samples...")
        sampleIndexes, testIndexes = randomPartition(sampleSize, data.shape[0])
        sampleInliers = data[sampleIndexes]
        # Check collinearity  
        while checkCollinearity(sampleInliers) : 
            print("Samples are collinear")
            print("Rechoosing samples...")
            sampleIndexes, testIndexes = randomPartition(sampleSize, data.shape[0])
            sampleInliers = data[sampleIndexes]
        print(f"Samples : \n{sampleInliers}\n")
        # Fit an ellipse 
        params = model.fit(sampleInliers)
        print("Params : ")
        print(f"  A    : {params[0]} \n  B    : {params[1]} \n  C    : {params[2]} \n  D    : {params[3]} \n  E    : {params[4]} \n  F    : [1] \n")        
        # Check if its an ellipse 
        if model.validateModel(params) :
            print("Ellipse : True")
            # Computing the characterisitcs of the ellipse 
            tempModel = model.computeModel(params)
            # Discard the pixels that has greater distance than the threshold 
            filteredIndexes = filterIndexes(data, testIndexes, threshold, tempModel)
            filteredInliers = data[filteredIndexes]
            inliersRatio = len(filteredInliers) / len(data) * 100
            # If there are 
            # more points that the tolerance then it is a good fit
            if inliersRatio > tolerance : 
                print("Inliers more than tolerance")
                allInliers = np.concatenate((filteredInliers, sampleInliers))
                error      = model.computeFittingError(allInliers, tempModel)
                if error < bestError : 
                    bestModel = tempModel
            else : print("Inliers less than tolerance")
        else : 
            print("Ellipse : False")

        iteration = iteration + 1
        print("================================================================")                    
        
    return bestModel

def checkCollinearity(data) : 

    comb   = list(combinations(data, 3))
    retval = False

    for i in comb : 
        if areCollinear(i[0], i[1], i[2]) :
            retval = True
    
    return retval

def areCollinear(x, y, z) : 

    retval = False
    p0     = x[0] * (y[1] - z[1])
    p1     = y[0] * (z[1] - x[1])
    p2     = z[0] * (x[1] - y[1])
    area   = p0 + p1 + p2

    if (area == 0) : 
        retval = True

    return retval

def randomPartition(sampleSize, dataRows) : 

    allIndexes   = np.arange(dataRows)
    np.random.shuffle(allIndexes)
    sampleIndexes = allIndexes[:sampleSize]
    testIndexes  = allIndexes[sampleSize:]

    return sampleIndexes, testIndexes    

def filterIndexes(data, indexes, threshold, model) :

    filteredIndexes = [] 

    for index in indexes : 
        temp = data[index]
        # dist = getDistance(temp, params[0][0], params[1][0], params[2][0], x0, y0)
        dist = getDistance(temp, model[0], model[1], model[2], model[3], model[4])
        if (dist < threshold) : 
            filteredIndexes.append(index)

    return filteredIndexes
