# CSC3022F 2021 ML Assignment 2
# Part 1: Value Iteration
# Author: WNGJIA001

# import packages
import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Animate import generateAnimat

# global variables
records = []
rewards = []
mines = []
policy = []
opt_pol = []
# initialise variables, to be updated with command line parsing
width = 0
height = 0
start_state = (0, 0)
end_state = (0, 0)
k = 0
g = 0.0

def parseCommandLine():
    """
    parse command line function:
    parse command line flags and arguments into variables
    """
    global mines, width, height, start_state, end_state, k, g
    # set command line options
    parser = argparse.ArgumentParser(description='Performs value iteration.')
    parser.add_argument('width', type=check_size)
    parser.add_argument('height', type=check_size)
    parser.add_argument('--start', '-start', nargs=2, type=positive_int, metavar=('xpos', 'ypos'), help='starting location of the agent')
    parser.add_argument('--end', '-end', nargs=2, type=positive_int, metavar=('xpos', 'ypos'), help='target destination of the agent')
    parser.add_argument('--k', '-k', default=3, type=positive_int, metavar='k', help='number of landmines')
    parser.add_argument('--gamma', '-gamma', default=0.8, type=check_gamma, metavar='g', help='discount factor')
    args = parser.parse_args()
    # assign argument values to variables
    width = args.width
    height = args.height
    if not args.start == None and args.start == args.end:
        sys.exit('error: start and end locations can not be the same')
    else:
        # assign starting location or generate random location if not passed
        if args.start == None:
            start_state = (random.randint(0, width-1), random.randint(0, height-1))
        elif args.start[0] < width and args.start[1] < height:
            start_state = tuple(args.start)
        else:
            sys.exit('invalid start values %s: defined outside the environment' % args.start)
        # assign target destination or generate random location if not passed
        if args.end == None:
            while True:
                end_state = (random.randint(0, width-1), random.randint(0, height-1))
                if not end_state == start_state: break
        elif args.end[0] < width and args.end[1] < height:
            end_state = tuple(args.end)
        else:
            sys.exit('invalid end values %s: defined outside the environment' % args.start)
    # assign k by checking k is less than width*height-2
    k = args.k
    if k <= width*height-2:
        for i in range(k):
            while True:
                mine = (random.randint(0, width-1), random.randint(0, height-1))
                if not (mine == start_state or mine == end_state or mine in mines): break
            mines.append(mine)
    else:
        sys.exit("invalid k value: '%s', should not exceed (width*height-2) = %s" % (k, width*height-2))
    # assign discount factor gamma
    g = args.gamma
    #print('w: ', width, ', h: ', height, ', s: ', start_state, ', e: ', end_state, ', k: ', k, ', g: ', g)
    #print(mines)

# helper functions to check if the parsed values are valid
def check_size(i):
    """
    check size funtion:
    check if the parsed value is a valid int value for width or height
    width and height should be greater than 1 in order to
    generate a meaningful environment
    """
    value = int(i)
    if value <= 1:
        raise argparse.ArgumentTypeError("invalid size value: '%s', should be greater than 1" % i)
    return value
def positive_int(i):
    """
    check positive integer function:
    check if the parsed value is a valid positive int value
    xpos , ypos and k should be integers greater than or equal to 0
    """
    value = int(i)
    if value < 0:
        raise argparse.ArgumentTypeError("invalid positive int number: '%s'" % i)
    return value
def check_gamma(i):
    """
    check gamma function:
    check if the parsed value is a valid value between 0 and 1
    the discount factor gamma should fall between 0 and 1
    """
    value = float(i)
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("invalid gamma value: '%s', should be between 0 and 1" % i)
    return value

def value_iteration(prev_record):
    """
    value iteration algorithm:
        input: record of values from previous iteration
        output: approximately optimal policy and value function for current iteration
    for each state, find the max value in taking any action in the four directions
    the value function and approximately optimal policy is returned
    """
    policy = []
    record = np.zeros((height, width))
    prev_state = start_state
    for row in range(height):
        pol = [] # a row of the policy matrix
        for column in range(width):
            # for each state, if terminal state: keep as reward value
            # else: find the max value for taking actions in any of 4 directions
            if (column, row) in mines or (column, row) == end_state:
                record[row][column] = rewards[row][column]
                pol.append((row, column))
            else:
                values = []
                pols = []
                r = row - 1 # UP
                if r >= 0 and r < height:
                    values.append(rewards[r][column]+g*prev_record[r][column])
                    pols.append((column, r))
                r = row + 1 # DOWN
                if r >= 0 and r < height:
                    values.append(rewards[r][column]+g*prev_record[r][column])
                    pols.append((column, r))
                c = column - 1 # LEFT
                if c >= 0 and c < width:
                    values.append(rewards[row][c]+g*prev_record[row][c])
                    pols.append((c, row))
                c = column + 1 # RIGHT
                if c >= 0 and c < width:
                    values.append(rewards[row][c]+g*prev_record[row][c])
                    pols.append((c, row))
                # find max value and policy/next state
                max_value = max(values)
                record[row][column] = max_value
                pol.append(pols[np.argmax(values)])
        policy.append(pol)
    return policy, record

def getOptimalPolicy():
    """
    get optimal policy function:
    iterate through policy function to find the optimal policy
    """
    opt_pol = [start_state]
    pol = start_state
    while not pol == end_state:
        pol = policy[pol[1]][pol[0]]
        opt_pol.append(pol)
    return opt_pol

def setReards():
    """
    set rewards function:
    set the reward function for all states
    """
    # set up rewards for all states: end_state=100, landmine=-100, else=-1
    rewards = np.zeros((height, width))
    # rewards.fill(-1)
    rewards[end_state[1]][end_state[0]] = 100
    for mine in mines:
        rewards[mine[1]][mine[0]] = -100
    return rewards

def convergence():
    """
    convergence function:
    check if the value iteration has converged
    """
    if len(records) >= 2 and records[-2] == records[-1]:
        return True
    return False

if __name__=='__main__':
    parseCommandLine()
    # set up rewards
    rewards = setReards()
    # initialise value function = 0 for all states
    record = np.zeros((height, width))
    # iterate while not convergence
    while not convergence():
        prev_record = record # if not records else records[-1]
        policy, record = value_iteration(prev_record)
        records.append(record.tolist())
    # print(np.round(records, 1))
    print(np.round(records[-1], 1))
    print(policy)
    opt_pol = getOptimalPolicy()
    print(opt_pol)
