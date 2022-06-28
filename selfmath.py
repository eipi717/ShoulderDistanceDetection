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

if __name__ == '__main__':
    Male_ = [34, 39, 38, 49, 37, 49, 41, 44] # mean = 41.375

    Male_12321 = [36, 49, 32, 48, 37, 30, 45, 38, 43, 35, 35, 42, 46, 37, 42] # mean = 39.666666666666664

    Female_12321 = [35, 40, 38, 35, 40, 40, 44, 44, 34, 35, 45, 32, 37] # mean = 38.38461538461539

    Female_ = [41, 54, 45, 51, 66, 56, 53, 40, 50, 44] # mean = 50

    Excel_Male_99 = [34.05263158] # mean = 34.05263158
    Excel_Female_99 = [38.89473684] # mean = 38.89473684

    Excel_Male_800 = [35] # mean = 35
    Excel_Female_800 = [35.6] # mean = 35.6

    Excel_Male_234432 = [37.55555556] # mean = 37.55555556
    Excel_Female_234432 = [38.83333333] # mean = 38.83333333

    total_male = len(Male_) + len(Male_12321) + 14 + 19 + 9
    total_female = len(Female_) + len(Female_12321) + 19 + 20 + 6

    Male = [Male_ + Male_12321 + Excel_Male_99 + Excel_Male_800 + Excel_Male_234432]
    Female = [Female_ + Female_12321 + Excel_Female_99 + Excel_Female_800 + Excel_Female_234432]


    print("\n******* Overall Mean ********")
    print("Male: \nMean: ", np.mean(Male), "\nSize: ", total_male)
    print("Female: \nMean: ", np.mean(Female), "\nSize: ", total_female)

    print("\n******* Overall std ********")
    print("Male: \nstd: ", np.std(Male), "\nSize: ", total_male)
    print("Female: \nstd: ", np.std(Female), "\nSize: ", total_female)
