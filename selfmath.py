# Find the angle between two lines
# Input: slopes m1, m2
# output: angle converted to degree

import numpy as np
from numpy.linalg import norm

def LinesInAngle(m1, m2):
    rad = np.arctan((m2 - m1) / (1 + (m1 * m2)))
    deg = rad * (180 / np.pi)
    return np.abs(deg)


def MidPt(x, y):
    MP = ((x[0] + y[0])/2, (x[1] + y[1])/2)
    return MP


# Find the slope of the line
# Input: points x1, x2
# Output: slope of the line

def GetSlope(x1, x2):
    y = x2[1] - x1[1]
    x = x2[0] - x1[0]

    if x == 0:
        return "Infinity!"
    else:
        return y / x


def dist(x, y):
    return norm(np.array(x)-np.array(y))