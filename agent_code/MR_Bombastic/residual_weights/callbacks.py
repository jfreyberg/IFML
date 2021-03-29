import numpy as np
import pickle
from settings import e
from settings import s
from random import shuffle
from agent_code.my_agent.feature_extraction import *
from agent_code.my_agent.algorithms import new_reward, q_gd_linapprox


#########################################################################

def setup(self):
    
    self.actions = [ 'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT' ]
    #self.train_mode = "Greed_batch_test"
    self.init_mode = "handpicked_init"
    #self.init_mode = 'init1'
    self.alpha_set = "const0.2"
    # load weights
    try:
        self.weights = np.load('agent_code/my_agent/models/.npy')
        #self.training_weights = np.load('train_weights_{}_{}_{}.npy'.format(self.train_mode, self.init_mode, self.alpha_set))
        #self.training_rewards = np.load('train_rewards_{}_{}_{}.npy'.format(self.train_mode, self.init_mode, self.alpha_set))
        print("weights loaded")
    except:
        self.weights = []
        self.training_weights = []
        self.training_rewards = []
        print("no weights found ---> create new weights")

    # Define Rewards
    self.total_R = 0
    self.reward_round = 0
    
    # Step size or gradient descent 
    self.alpha = 0.2
    self.gamma = 0.95
    self.EPSILON = 0.2
    self.round = 1
    
    self.history = []
    
    
#####################################################################

def act(self):
    
    
    """
    actions order: 'UP', 'DOWN', LEFT', 'RIGHT', 'BOMB', 'WAIT'    
    """
    
    # load state 
    game_state = self.game_state 
    #print("step {}".format(game_state['step']))

    # Compute features state 
    F = RLFeatureExtraction(self.game_state)
    feature_state = F.state()
    self.prev_state = feature_state
    
    #different initial guesses can be defined here: 
    if len(self.weights) == 0:
        print('no weights, init weights')
        if self.init_mode == 'handpicked_init':
            self.weights =  np.array([-15, 1.5, -81, -2, 10, -1, 3.5, 1.7, 0.8, 0.8, 1.5, 2, 13, -10, 2, 2])
        elif self.init_mode == 'init1':
            self.weights = np.ones(feature_state.shape[1])  
        elif self.init_mode == 'initRand':
            self.weights = np.random.rand(self.prev_state.shape[1])
    
    #print(self.weights)
    self.logger.info('Pick action')
    
    # Linear approximation approach
    q_approx = np.dot(feature_state, self.weights)    
    best_actions = np.where(q_approx == np.max(q_approx))[0] 
    shuffle(best_actions)

    q_next_action = self.actions[best_actions[0]] #GREEDY POLICY
    self.next_action = q_next_action
    self.prev_action = self.next_action
    
    #print("q action picked  ", q_next_action)
    
    
#####################################################################

def reward_update(self):

    self.logger.info('IN TRAINING MODE ')
    if self.game_state['step']>1:
        #print('LEARNING')
        
        reward = new_reward(self.events)
        self.total_R += reward      
        self.reward_round += reward
        
        hist_entry = {'state': self.prev_state, 'action': self.prev_action, 'reward':reward}
        self.history.append(hist_entry)
        
        #print(hist_entry)
           
#####################################################################

def end_of_episode(self):
       
    print("end of round {}".format(self.round))
    self.round += 1
    #reset alpha for next round
    self.alpha = 0.2
    
    ## calculate new rewards 
    reward = new_reward(self.events)
    self.total_R += reward 
    self.reward_round += reward
    

            
    hist_entry = {'state': self.prev_state, 'action': self.prev_action, 'reward':reward}
    self.history.append(hist_entry)

    ################ LEARNING    

    print('total reward from this round: {}'.format(self.reward_round))
    print('total r: {}'.format(self.total_R))
    self.reward_round = 0    
    
    temp_weights  = np.zeros(14)

    for i in range(len(self.history)-1):

        weights = self.weights
        prev_state = self.history[i]['state']
        prev_action = self.history[i]['action']
        next_state = self.history[i+1]['state']
        next_action = self.history[i+1]['action']

        prev_sa = prev_state[self.actions.index(prev_action),:]
        next_sa = next_state[self.actions.index(next_action),:]
        
        temp = (reward + self.gamma * np.dot(next_sa,weights) - np.dot(prev_sa,weights))             * (np.dot(prev_sa,weights)-np.dot(next_sa, weights))* weights

        temp_weights += temp

    temp_weights = (-self.alpha) * (temp_weights/len(self.history))
    print('new weights = {}'.format(weights + temp_weights))
    self.weights += temp_weights
    self.weights /= np.sum(self.weights)

   

    ############## SAVING LEARNING FROM ONE EPISODE 
    #np.save('weights_{}_{}_{}.npy'.format(self.train_mode, self.init_mode, self.alpha_set), self.weights)

    self.training_weights = np.append(self.training_weights, self.weights)
    #np.save('train_weights_{}_{}_{}.npy'.format(self.train_mode, self.init_mode, self.alpha_set), self.training_weights)
    
    self.training_rewards = np.append(self.training_rewards, self.total_R)
    #np.save('train_rewards_{}_{}_{}.npy'.format(self.train_mode, self.init_mode, self.alpha_set), self.training_rewards)


    ################# RESET PARAMETERS FOR NEXT ROUND
    if self.round%10 == 0:
        self.total_R = 0
    

