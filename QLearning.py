# CSC3022F 2021 ML Assignment 2
# Part 2: Q Learning
# Author: WNGJIA001

# importing libraries
import sys
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Animate import generateAnimat
from Helper import getOptPol

# global variables
Q_values = [] # Q matrix of each epoch
records = [] # records for animation generation
rewards = [] # reward function
mines = [] # landmines
opt_pol = [] # optimal policy
visits = [] # visits matrix
# initialise variables, to be updated with command line parsing
width = 0
height = 0
start_state = (0, 0)
end_state = (0, 0)
k = 0
g = 0.0
l = -1.0
e = 0

def parseCommandLine():
    """
    parse command line function:
    parse command line flags and arguments into variables
    """
    global mines, width, height, start_state, end_state, k, g, l, e
    # set command line options
    parser = argparse.ArgumentParser(description='Performs Q Learning.')
    parser.add_argument('width', type=check_size)
    parser.add_argument('height', type=check_size)
    parser.add_argument('-start', '--start', nargs=2, type=positive_int, metavar=('xpos', 'ypos'), help='starting location of the agent')
    parser.add_argument('-end','--end',  nargs=2, type=positive_int, metavar=('xpos', 'ypos'), help='target destination of the agent')
    parser.add_argument('-k', '--k', default=3, type=positive_int, metavar='k', help='number of landmines')
    parser.add_argument('-gamma', '--gamma', default=0.8, type=check_gamma, metavar='g', help='discount factor')
    parser.add_argument('-learning', '--learning', type=check_gamma, metavar='l', help='learning rate')
    parser.add_argument('-epochs', '--epochs', default=500, type=positive_int, metavar='e', help='number of episodes the agent should learn for')
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
    if not args.learning == None:
        l = args.learning
    e = args.epochs

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

def initialise():
    """
    set rewards function:
    set the reward function for all states
    """
    # set up rewards for all states: end_state=100, landmine=-100, else=0
    rewards = np.zeros((height, width))
    rewards[end_state[1]][end_state[0]] = 100
    for mine in mines:
        rewards[mine[1]][mine[0]] = -100
    visits = np.zeros((width*height, 4))
    record = np.zeros((height, width))
    return rewards, visits, record

def qLearning(prev_Q):
    """
    Q learning algorithm:
        input: record of Q values from previous iteration
        output: approximately optimal policy and Q matrix from current iteration
    take a random state, loop until reaching end state, updating the Q values for
    all the states passed through
    """
    # deep copy previous Q matrix for updating
    curr_Q = copy.deepcopy(prev_Q)
    # select a random initial current state
    curr_state = (random.randint(0, width-1), random.randint(0, height-1))
    while not (curr_state in mines or curr_state == end_state):
        # select a random future possible state
        next_state, action = doRandomAction(curr_state)
        max_value = 0
        # given current action, get maximum Q value for a next state
        next_row = next_state[1]
        next_column = next_state[0]
        if next_state in mines or next_state == end_state:
            curr_Q[next_row][next_column] = rewards[next_row][next_column]
        else:
            values = []
            r = next_row - 1 # UP
            if r >= 0 and r < height: values.append(prev_Q[r][next_column])
            r = next_row + 1 # DOWN
            if r >= 0 and r < height: values.append(prev_Q[r][next_column])
            c = next_column - 1 # LEFT
            if c >= 0 and c < width: values.append(prev_Q[next_row][c])
            c = next_column + 1 # RIGHT
            if c >= 0 and c < width: values.append(prev_Q[next_row][c])
            # find max value and policy/next state
            max_value = max(values)
        curr_row = curr_state[1]
        curr_column = curr_state[0]
        visits[curr_row*width+curr_column][action] += 1
        if l == -1.0:
            lrt = 1/(1+visits[curr_row*width+curr_column][action]) # decaying learning rate
        else:
            lrt = l # use user defined learning rate
        old_value = prev_Q[curr_row][curr_column]
        curr_Q[curr_row][curr_column] = old_value+lrt*(rewards[next_row][next_column]+g*max_value-old_value)
        curr_state = next_state
    return curr_Q

def doRandomAction(c_state):
    """
    get random action:
        input: current state position
        output: random next state, action taken
    given the current state, find a random action in any of the four possible
    directions and return the action and the position of next state
    """
    global visits
    while True:
        direction = random.randint(0, 3)
        if direction == 0: # UP
            y = c_state[1] - 1
            if y >= 0 and y < height:
                n_state = (c_state[0], y)
                break
        elif direction == 1: # DOWN
            y = c_state[1] + 1
            if y >= 0 and y < height:
                n_state = (c_state[0], y)
                break
        elif direction == 2:  # LEFT
            x = c_state[0] - 1
            if x >= 0 and x < width:
                n_state = (x, c_state[1])
                break
        elif direction == 3:  # RIGHT
            x = c_state[0] + 1
            if x >= 0 and x < width:
                n_state = (x, c_state[1])
                break
    return n_state, direction

if __name__=='__main__':
    parseCommandLine()
    # initialise
    rewards, visits, record = initialise()
    # loop through the number of epochs
    for i in range(e):
        prev_record = record # if not records else records[-1]
        record = qLearning(prev_record)
        Q_values.append(record.tolist())
    # get optimal policy
    opt_pol = getOptPol(Q_values[-1], start_state, end_state)
    # generate animation
    # reduce the number records if too many epochs
    if e < 60:
        records = np.round(Q_values, 1)
    else:
        ratio = int(e/60)
        for i in range(0, e, ratio):
            records.append(np.round(Q_values[i], 1))
    anim = generateAnimat(records, start_state, end_state, mines=mines, opt_pol=opt_pol,
        start_val=-10, end_val=100, mine_val=150, just_vals=False, generate_gif=True)
    plt.show()
