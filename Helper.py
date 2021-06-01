# CSC3022F 2021 ML Assignment 2
# Helper functions
# Author: WNGJIA001

import numpy as np

def getOptPol(record, start_state, end_state):
    """
    get optimal policy function:
    iterate through policy function to find the optimal policy
    """
    width = len(record[0])
    height = len(record)
    opt_pol = [start_state]
    pol = start_state
    while not pol == end_state:
        # find the next state or position with a max value
        values = []
        pols = []
        row = pol[1]
        column = pol[0]
        r = row - 1 # UP
        if r >= 0 and r < height:
            values.append(record[r][column])
            pols.append((column, r))
        r = row + 1 # DOWN
        if r >= 0 and r < height:
            values.append(record[r][column])
            pols.append((column, r))
        c = column - 1 # LEFT
        if c >= 0 and c < width:
            values.append(record[row][c])
            pols.append((c, row))
        c = column + 1 # RIGHT
        if c >= 0 and c < width:
            values.append(record[row][c])
            pols.append((c, row))
        max_value = max(values)
        pol = pols[np.argmax(values)]
        opt_pol.append(pol)
    return opt_pol
