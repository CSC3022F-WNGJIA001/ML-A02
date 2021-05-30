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
    value = int(i)
    if value <= 1:
        raise argparse.ArgumentTypeError("invalid size value: '%s', should be greater than 1" % i)
    return value
def positive_int(i):
    value = int(i)
    if value < 0:
        raise argparse.ArgumentTypeError("invalid positive int number: '%s'" % i)
    return value
def check_gamma(i):
    value = float(i)
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("invalid gamma value: '%s', should be between 0 and 1" % i)
    return value



def setReards():
    # set up rewards for all states: end_state=100, landmine=-100, else=-1
    rewards = np.zeros((height, width))
    # rewards.fill(-1)
    rewards[end_state[1]][end_state[0]] = 100
    for mine in mines:
        rewards[mine[1]][mine[0]] = -100
    return rewards

def convergence():
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
