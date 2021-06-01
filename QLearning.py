# CSC3022F 2021 ML Assignment 2
# Part 2: Q Learning
# Author: WNGJIA001

# importing libraries
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
l = 0.0
e = 0

def parseCommandLine():
    """
    parse command line function:
    parse command line flags and arguments into variables
    """
    global mines, width, height, start_state, end_state, k, g, l, e
    # set command line options
    parser = argparse.ArgumentParser(description='Performs value iteration.')
    parser.add_argument('width', type=check_size)
    parser.add_argument('height', type=check_size)
    parser.add_argument('--start', '-start', nargs=2, type=positive_int, metavar=('xpos', 'ypos'), help='starting location of the agent')
    parser.add_argument('--end', '-end', nargs=2, type=positive_int, metavar=('xpos', 'ypos'), help='target destination of the agent')
    parser.add_argument('--k', '-k', default=3, type=positive_int, metavar='k', help='number of landmines')
    parser.add_argument('--gamma', '-gamma', default=0.8, type=check_gamma, metavar='g', help='discount factor')
    parser.add_argument('--lrate', '-lrate', default=1.0, type=check_gamma, metavar='l', help='learning rate')
    parser.add_argument('--epochs', '-epochs', default=100, type=positive_int, metavar='e', help='number of episodes the agent should learn for')
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
    e = args.epochs
    l = args.lrate

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



if __name__=='__main__':
    parseCommandLine()
    print('w: ', width, ', h: ', height, ', s: ', start_state, ', e: ', end_state, ', k: ', k, ', g: ', g, ', l: ', l, ', e: ', e)
    print(mines)
