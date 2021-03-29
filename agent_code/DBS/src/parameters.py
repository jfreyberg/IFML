# global parameters
import os
import json

#N = 4  # how many fields to view in each direction
N = 3  # how many fields to view in each direction
#C = 6  # how many channels
C = 5  # how many channels
E = 19+6*2  # how many channels
OPTIONS = ['RIGHT', 'LEFT', 'BOMB', 'UP', 'DOWN', 'WAIT']
INVERSE_OPTIONS = {'RIGHT': 0, 'LEFT': 1, 'BOMB': 2, 'UP': 3, 'DOWN': 4, 'WAIT': 5}


def get_parameters():
    global N

    CONF = 'confs/default.json'

    conf = json.load(open(CONF, 'r'))

    if "N" in conf:
        N = conf['N']

    AGENT_CONFIG = conf['agent']
    AGENT_CONFIG["input_dim"] = (C, N*2+1, N*2+1)
    AGENT_CONFIG["extra_dim"] = E
    AGENT_CONFIG["action_dim"] = len(OPTIONS)

    if "AGENT_LOAD" in os.environ:
        AGENT_CONFIG['load_model'] = bool(os.environ['AGENT_LOAD'] != '0')
    if "AGENT_EVAL" in os.environ:
        AGENT_CONFIG['evaluate_model'] = bool(os.environ['AGENT_EVAL'] != '0')

    REWARDS = None

    return AGENT_CONFIG, REWARDS, N
