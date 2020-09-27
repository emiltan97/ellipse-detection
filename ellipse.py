import numpy as np

from math import sqrt, sin, cos
from utils import getDistance

class Ellipse : 

    def fit(self, data) :
        # P matrix 
        # [
        #  [x1**2 2x1y1 y1**2 2x1 2y1],
        #  [x2**2 2x2y2 y2**2 2x2 2y2], 
        #  [x3**2 2x3y3 y3**2 2x3 2y3],
        #  [x4**2 2x4y4 y4**2 2x4 2y4],
        #  [x5**2 2x5y5 y5**2 2x5 2y5]
        # ]
        
        # B matrix
        # [
            # [1],
            # [1],
            # [1],
            # [1],
            # [1]
        # ]

        # A = inverse P matrix dot multiply with B matrix

        col0 = np.vstack(np.square(data[:, 0]))
        col1 = np.vstack(np.square(data[:, 1]))
        col2 = np.vstack(2 * data[:, 0] * data[:, 1])
        col3 = np.vstack(2 * data[:, 0])
        col4 = np.vstack(2 * data[:, 1])

        P    = np.hstack(( col0, col1, col2, col3, col4 ))
        B    = np.array([ [-1], [-1], [-1], [-1], [-1] ])
        A    = np.linalg.inv(P) @ B

        return A

    def computeModel(self, params) :

        A = params[0][0]
        B = params[1][0]
        C = params[2][0]
        D = params[3][0]
        E = params[4][0]

        V = np.array([
            [A, C],
            [C, B]
        ])
        U = np.array([
            [D],
            [E]
        ])
        
        eigval, eigvec = np.linalg.eig(V) 
        theta  = self.computeAngle(eigvec)
        x0, y0 = self.computeCenter(V, U)
        a, b   = self.computeAxes(eigval, A, B, C, x0, y0, theta)

        return x0, y0, a, b, theta

    def validateModel(self, params) : 
            
        retval = False 
        A      = params[0][0]
        B      = params[1][0]
        C      = params[2][0] 

        if A > 0 and B > 0 and A*B-C**2 > 0 : 
            retval = True

        return retval

    def computeFittingError(self, data, model) : 

        totalDist = 0 

        for data in data : 
            dist      = getDistance(data, model[0], model[1], model[2], model[3], model[4])
            totalDist = totalDist + dist

        return totalDist

    def computeAxes(self, eigval, A, B, C, x0, y0, theta) : 

        lambda1 = eigval[0]
        lambda2 = eigval[1]

        G = np.array([
            [x0, y0]
        ]) @ np.array([
            [A, C], 
            [C, B]
        ]) @ np.array([
            [x0],
            [y0]
        ]) - 1

        mat1 = np.array([
            [A/G[0][0], C/G[0][0]],
            [C/G[0][0], B/G[0][0]]
        ])
        mat2 = np.array([
            [ cos(theta), sin(theta)],
            [-sin(theta), cos(theta)]  
        ])
        mat3 = np.array([
            [cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]
        ])
        
        prd = np.linalg.inv(mat2) @ np.linalg.inv(mat3) @ mat1 
        a = sqrt(1 / prd[0][0])
        b = sqrt(1 / prd[1][1])

        if a < b : 
            temp = a 
            a    = b
            b    = temp

        return a, b

    def computeAngle(self, eigvec) :

        Z = complex(-eigvec[1, 1], -eigvec[0, 1])
        theta = np.angle(Z)

        return theta

    def computeCenter(self, mat1, mat2) :

        mat  = -(np.linalg.inv(mat1) @ mat2)
        x0   = mat[0][0]
        y0   = mat[1][0]

        return x0, y0