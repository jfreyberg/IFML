import numpy as np
import copy


'''
computes the manhatten distance between point p1 and p2
'''


def manhatten_distance(p1, p2):
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])


