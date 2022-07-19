'''
## Train ##
# Code to train D4PG Network on OpenAI Gym environments
@author: Mark Sinton (msinto93@gmail.com) 
'''
import threading
import random
import tensorflow as tf
import numpy as np

from params import train_params
from utils.prioritised_experience_replay import PrioritizedReplayBuffer   
from utils.gaussian_noise import GaussianNoiseGenerator
from agent import Agent
from learner import Learner
    
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()     

import uvicorn

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from api.route import KyonRL_setup

from collections import deque



class Recommend_core():
    def __init__(self):

        tf.reset_default_graph()
        
        # Set random seeds for reproducability
        np.random.seed(train_params.RANDOM_SEED)
        random.seed(train_params.RANDOM_SEED)
        tf.set_random_seed(train_params.RANDOM_SEED)
        
        # Initialise prioritised experience replay memory
        self.PER_memory = PrioritizedReplayBuffer(train_params.REPLAY_MEM_SIZE, train_params.PRIORITY_ALPHA)
        # Initialise Gaussian noise generator
        gaussian_noise = GaussianNoiseGenerator(train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.NOISE_SCALE)
                
        # Create session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)  
        
        # Create threads for learner process and agent processes       
        threads = []
        # Create threading events for communication and synchronisation between the learner and agent threads
        self.run_agent_event = threading.Event()
        stop_agent_event = threading.Event()
        
        # with tf.device('/device:GPU:0'):
        # Initialise learner
        learner = Learner(sess, self.PER_memory, self.run_agent_event, stop_agent_event)
        # Build learner networks
        learner.build_network()
        # Build ops to update target networks
        learner.build_update_ops()
        # Initialise variables (either from ckpt file if given, or from random)
        learner.initialise_vars()
        # Get learner policy (actor) network params - agent needs these to copy latest policy params periodically
        learner_policy_params = learner.actor_net.network_params + learner.actor_net.bn_params
        
        threads.append(threading.Thread(target=learner.run_api))
        

        # Initialise agent
        self.agent = Agent(sess, train_params.ENV, train_params.RANDOM_SEED)
        # Build network
        self.agent.build_network(training=True)
        # Build op to periodically update agent network params from learner network
        self.agent.build_update_op(learner_policy_params)
        # Create Tensorboard summaries to save episode rewards
        if train_params.LOG_DIR is not None:
            self.agent.build_summaries(train_params.LOG_DIR + ('/agent_%02d' % 1))

        # Perform initial copy of params from learner to agent
        sess.run(self.agent.update_op)

        # Initialise var for logging episode reward
        if train_params.LOG_DIR is not None:
            sess.run(self.agent.init_reward_var)

        # Initially set threading event to allow agent to run until told otherwise
        self.run_agent_event.set()



        # threads.append(threading.Thread(target=agent.run, args=(PER_memory, gaussian_noise, run_agent_event, stop_agent_event)))
        
        for t in threads:
            t.start()
            
        # for t in threads:
        #     t.join()
    
    def get_learningPoint(self, student_ID, subject, history_topic, history_action, masteries, history_score):
        return self.agent.inference(student_ID, subject, history_topic, self.PER_memory, self.run_agent_event, history_action, masteries, history_score)

    # def  get_learningPoint()
# if  __name__ == '__main__':
    # train()         
            
        
    
    
        
        