'''
## Agent ##
# Agent class - the agent explores the environment, collecting experiences and adding them to the PER buffer. Can also be used to test/run a trained network in the environment.
@author: Mark Sinton (msinto93@gmail.com) 
'''

from math import fabs
import os
import sys
import tensorflow as tf
import numpy as np
import scipy.stats as ss
from collections import deque
import cv2
# import imageio
import threading
import json
import random
from params import train_params, test_params, play_params
from utils.network import Actor, Actor_BN
from utils.env_wrapper import PendulumWrapper, LunarLanderContinuousWrapper, BipedalWalkerWrapper

from env_kyon import SimStudent



import tensorflow.compat.v1 as tf
from variables_old import LP_SEGMENT
tf.disable_v2_behavior() 

from utils_ import topic_recommender, mask_others_lp_not_in_topic, load_deque, save_deque, read_masteries, save_masteries

class Agent:
  
    def __init__(self, sess, env, seed, n_agent=0):
        print("Initialising agent %02d... \n" % n_agent)
         
        self.sess = sess        
        self.n_agent = n_agent
        self.lock = threading.Lock()
        self.num_steps = 0
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        
       
        # Create environment    
        if env == 'Pendulum-v0':
            self.env_wrapper = PendulumWrapper()
        elif env == 'LunarLanderContinuous-v2':
            self.env_wrapper = LunarLanderContinuousWrapper()
        elif env == 'BipedalWalker-v2':
            self.env_wrapper = BipedalWalkerWrapper()
        elif env == 'BipedalWalkerHardcore-v2':
            self.env_wrapper = BipedalWalkerWrapper(hardcore=True)
        elif env == 'kyon':
            self.env_wrapper = SimStudent()
        else:
            raise Exception('Chosen environment does not have an environment wrapper defined. Please choose an environment with an environment wrapper defined, or create a wrapper for this environment in utils.env_wrapper.py')
        # self.env_wrapper.set_random_seed(seed*(n_agent+1))
              
    def build_network(self, training):
        # Input placeholder    
        self.state_ph = tf.placeholder(tf.float32, ((None,) + train_params.STATE_DIMS)) 
        
        if training:
            # each agent has their own var_scope
            var_scope = ('actor_agent_%02d'%self.n_agent)
        else:
            # when testing, var_scope comes from main learner policy (actor) network
            var_scope = ('learner_actor_main')
          
        # Create policy (actor) network
        if train_params.USE_BATCH_NORM:
            self.actor_net = Actor_BN(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, is_training=False, scope=var_scope)
            self.agent_policy_params = self.actor_net.network_params + self.actor_net.bn_params
        else:
            self.actor_net = Actor(self.state_ph, train_params.STATE_DIMS, train_params.ACTION_DIMS, train_params.ACTION_BOUND_LOW, train_params.ACTION_BOUND_HIGH, train_params.DENSE1_SIZE, train_params.DENSE2_SIZE, train_params.FINAL_LAYER_INIT, scope=var_scope)
            self.agent_policy_params = self.actor_net.network_params
                        
    def build_update_op(self, learner_policy_params):
        # Update agent's policy network params from learner
        update_op = []
        from_vars = learner_policy_params
        to_vars = self.agent_policy_params
                
        for from_var,to_var in zip(from_vars,to_vars):
            update_op.append(to_var.assign(from_var))
        
        self.update_op = update_op
                        
    def build_summaries(self, logdir):
        # Create summary writer to write summaries to disk
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.summary_writer = tf.summary.FileWriter(logdir, self.sess.graph)
        
        # Create summary op to save episode reward to Tensorboard log
        self.ep_reward_var = tf.Variable(0.0, trainable=False, name=('ep_reward_agent_%02d'%self.n_agent))
        tf.summary.scalar("Episode Reward", self.ep_reward_var)
        self.summary_op = tf.summary.merge_all()
        
        # Initialise reward var - this will not be initialised with the other network variables as these are copied over from the learner
        self.init_reward_var = tf.variables_initializer([self.ep_reward_var])
            
    def run(self, PER_memory, gaussian_noise, run_agent_event, stop_agent_event):
        # Continuously run agent in environment to collect experiences and add to replay memory
                
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()
        
        # Perform initial copy of params from learner to agent
        self.sess.run(self.update_op)
        
        # Initialise var for logging episode reward
        if train_params.LOG_DIR is not None:
            self.sess.run(self.init_reward_var)
        
        # Initially set threading event to allow agent to run until told otherwise
        run_agent_event.set()
        
        num_eps = 0
        
        while not stop_agent_event.is_set():
            num_eps += 1
            # Reset environment and experience buffer
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            self.exp_buffer.clear()
            
            num_steps = 0
            episode_reward = 0
            ep_done = False
            topics_done = 0
            
            while not ep_done:
                num_steps += 1
                ## Take action and store experience
                if train_params.RENDER:
                    self.env_wrapper.render()
                action = self.sess.run(self.actor_net.output , {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                action_prev = np.where(action == np.amax(action))[0]
                action += (gaussian_noise() * train_params.NOISE_DECAY**num_eps)
                # action = self.random_zeros_location(state=state)
                action_ = np.where(action == np.amax(action))[0]
                next_state, reward, terminal, curr_topic = self.env_wrapper.step(action, num_steps == train_params.MAX_EP_LENGTH)
                
                episode_reward += reward 
                               
                next_state = self.env_wrapper.normalise_state(next_state)
                reward = self.env_wrapper.normalise_reward(reward)
                
                self.exp_buffer.append((state, action, reward))
                
                # We need at least N steps in the experience buffer before we can compute Bellman rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= train_params.N_STEP_RETURNS:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = train_params.DISCOUNT_RATE
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= train_params.DISCOUNT_RATE
                    
                    # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                    run_agent_event.wait()   
                    PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)

                if terminal:
                    next_state = self.env_wrapper.set_topicDone(curr_topic)

                state = next_state
                
                if terminal or num_steps == train_params.MAX_EP_LENGTH:
                    # Log total episode reward
                    if train_params.LOG_DIR is not None:
                        summary_str = self.sess.run(self.summary_op, {self.ep_reward_var: episode_reward})
                        self.summary_writer.add_summary(summary_str, num_eps)
                    # Compute Bellman rewards and add experiences to replay memory for the last N-1 experiences still remaining in the experience buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = train_params.DISCOUNT_RATE
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= train_params.DISCOUNT_RATE
                        
                        # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
                        run_agent_event.wait()     
                        PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)
                    
                    # Start next episode if all topic was passed
                    ep_done = self.is_TopicsDone(state)
                
            # Update agent networks with learner params every 'update_agent_ep' episodes
            if num_eps % train_params.UPDATE_AGENT_EP == 0:
                self.sess.run(self.update_op)
        
        # self.env_wrapper.close()

    
    def inference(self, student_ID, subject, history_topic, PER_memory, run_agent_event, history_action, curr_masteries, history_score):
        
        '''history_action : dict{topic1:action1, topic_1:action2, topic2:action1,..
           masteries: [masteries1, masteries2,..]'''

        # while not stop_agent_event.is_set():
        # self.lock.acquire()
        num_steps = len(history_action)
        # self.lock.release()

        episode_reward = 0

        exp_buffer = load_deque(student_ID)

        # Preprocess input

        
        try:
            curr_topic = history_topic[-1]
            old_action = history_action[-1]
            # history_action = json.loads(history_action)
            # history_action
            # curr_topic_ = list(history_action)[-1] # depend on history action
            # old_action = np.array([history_action[curr_topic]], dtype=np.float32) # depend on history action
            # prev_state = mask_others_lp_not_in_topic(masteries[-2], curr_topic)
        except:
            curr_topic = None
            old_action = None
            # prev_state = None

        try:
            prev_masteries = read_masteries(student_ID)
            # prev_topic = list(history_action)[-2]
            prev_state = mask_others_lp_not_in_topic(prev_masteries, curr_topic)
        except:
            prev_state = None

        save_masteries(student_ID, curr_masteries)

        curr_topic = topic_recommender(curr_masteries, curr_topic)
        state = mask_others_lp_not_in_topic(curr_masteries, curr_topic)
        
        ## Take action and store experience
        action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
        reward, terminal = self.env_wrapper.step_api(curr_topic, history_action, prev_state, history_score)
        
        episode_reward += reward 
                        
        next_state = self.env_wrapper.normalise_state(state)
        reward = self.env_wrapper.normalise_reward(reward)
        
        if prev_state is not None:
            exp_buffer.append((prev_state, np.array([old_action], dtype=np.float32), reward))
            
        
        # We need at least N steps in the experience buffer before we can compute Bellman rewards and add an N-step experience to replay memory
        if len(exp_buffer) >= train_params.N_STEP_RETURNS:
            state_0, action_0, reward_0 = exp_buffer.popleft()
            discounted_reward = reward_0
            gamma = train_params.DISCOUNT_RATE
            for (_, _, r_i) in exp_buffer:
                discounted_reward += r_i * gamma
                gamma *= train_params.DISCOUNT_RATE
            
            # If learner is requesting a pause (to remove samples from PER), wait before adding more samples
            run_agent_event.wait()   
            PER_memory.add(state_0, action_0, discounted_reward, next_state, terminal, gamma)
        
        if prev_state is not None:
            save_deque(student_ID, exp_buffer)

        # Update agent networks with learner params every 'update_agent_ep' episodes

        if num_steps % train_params.UPDATE_AGENT_EP == 0:
            self.sess.run(self.update_op)
        
        return student_ID, self.mapping_action(action, state), curr_topic

    def mapping_action(self, action, state ):
        result = action
        if state[int(action)] == 1.0:
            list_zeroIndex = np.where(state == 0.0)[0]
            temp = 1000
            for i in list_zeroIndex:
                delta = abs(int(action)- i)
                if delta < temp :
                    temp = delta
                    result = np.array([i], dtype=np.float32)

        return result

    def random_zeros_location(self, state):
        list_zeroIndex = np.where(state == 0.0)[0]
        return np.array([list_zeroIndex[random.randint(0,len(list_zeroIndex)-1)]],dtype=np.float32)


    def get_expBuffer(self, student_ID)->deque():
        return None

    def is_TopicsDone(self, state):
        '''if all the value of state is 1 => all learning point was , vice versa'''
        for value in state:
            if value < 1:
                return False
        return True

    def load_checkpoint(self):
        def load_ckpt(ckpt_dir, ckpt_file):
            # Load ckpt given by ckpt_file, or else load latest ckpt in ckpt_dir
            loader = tf.train.Saver()    
            if ckpt_file is not None:
                ckpt = ckpt_dir + '/' + ckpt_file  
            else:
                ckpt = tf.train.latest_checkpoint(ckpt_dir)
             
            loader.restore(self.sess, ckpt)
            sys.stdout.write('%s restored.\n\n' % ckpt)
            sys.stdout.flush() 
             
            ckpt_split = ckpt.split('-')
            self.train_ep = ckpt_split[-1]
        
        # Load ckpt from ckpt_dir
        load_ckpt(train_params.CKPT_DIR, train_params.CKPT_FILE)
    
    def test(self):   
        # Test a saved ckpt of actor network and save results to file (optional)
        
        def load_ckpt(ckpt_dir, ckpt_file):
            # Load ckpt given by ckpt_file, or else load latest ckpt in ckpt_dir
            loader = tf.train.Saver()    
            if ckpt_file is not None:
                ckpt = ckpt_dir + '/' + ckpt_file  
            else:
                ckpt = tf.train.latest_checkpoint(ckpt_dir)
             
            loader.restore(self.sess, ckpt)
            sys.stdout.write('%s restored.\n\n' % ckpt)
            sys.stdout.flush() 
             
            ckpt_split = ckpt.split('-')
            self.train_ep = ckpt_split[-1]
        
        # Load ckpt from ckpt_dir
        load_ckpt(test_params.CKPT_DIR, test_params.CKPT_FILE)
        
        # Create Tensorboard summaries to save episode rewards
        if test_params.LOG_DIR is not None:
            self.build_summaries(test_params.LOG_DIR)
            
        rewards = [] 

        for test_ep in range(1, test_params.NUM_EPS_TEST+1):
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            ep_reward = 0
            step = 0
            ep_done = False
            
            while not ep_done:
                if test_params.RENDER:
                    self.env_wrapper.render()
                temp = np.genfromtxt('./input.txt', dtype=np.float64)
                action  = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0] # Add batch dimension to single state input, and remove batch dimension from single action output
                action = self.mapping_action(action,state)
                state, reward, terminal, current_topic = self.env_wrapper.step(action, step == test_params.MAX_EP_LENGTH)
                
                state = self.env_wrapper.normalise_state(state)
                
                ep_reward += reward
                step += 1
                 
                # Episode can finish either by reaching terminal state or max episode steps
                if terminal or step == test_params.MAX_EP_LENGTH:
                    sys.stdout.write('\x1b[2K\rTest episode {:d}/{:d}'.format(test_ep, test_params.NUM_EPS_TEST))
                    sys.stdout.flush()   
                    rewards.append(ep_reward)
                    ep_done = True   
                
        mean_reward = np.mean(rewards)
        error_reward = ss.sem(rewards)
                
        sys.stdout.write('\x1b[2K\rTesting complete \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(mean_reward, error_reward))
        sys.stdout.flush()  
        
        # Log average episode reward for Tensorboard visualisation
        if test_params.LOG_DIR is not None:
            summary_str = self.sess.run(self.summary_op, {self.ep_reward_var: mean_reward})
            self.summary_writer.add_summary(summary_str, self.train_ep)
         
        # Write results to file        
        if test_params.RESULTS_DIR is not None:
            if not os.path.exists(test_params.RESULTS_DIR):
                os.makedirs(test_params.RESULTS_DIR)
            output_file = open(test_params.RESULTS_DIR + '/' + test_params.ENV + '.txt' , 'a')
            output_file.write('Training Episode {}: \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(self.train_ep, mean_reward, error_reward))
            output_file.flush()
            sys.stdout.write('Results saved to file \n\n')
            sys.stdout.flush()      
        
        self.env_wrapper.close()       
        
    def play(self):   
        # Play a saved ckpt of actor network in the environment, visualise performance on screen and save a GIF (optional)
        
        def load_ckpt(ckpt_dir, ckpt_file):
            # Load ckpt given by ckpt_file, or else load latest ckpt in ckpt_dir
            loader = tf.train.Saver()    
            if ckpt_file is not None:
                ckpt = ckpt_dir + '/' + ckpt_file  
            else:
                ckpt = tf.train.latest_checkpoint(ckpt_dir)
        
            loader.restore(self.sess, ckpt)
            sys.stdout.write('%s restored.\n\n' % ckpt)
            sys.stdout.flush() 
             
            ckpt_split = ckpt.split('-')
            self.train_ep = ckpt_split[-1]
        
        # Load ckpt from ckpt_dir
        load_ckpt(play_params.CKPT_DIR, play_params.CKPT_FILE)
        
        # Create record directory
        if not os.path.exists(play_params.RECORD_DIR):
            os.makedirs(play_params.RECORD_DIR)

        for ep in range(1, play_params.NUM_EPS_PLAY+1):
            state = self.env_wrapper.reset()
            state = self.env_wrapper.normalise_state(state)
            step = 0
            ep_done = False
            
            while not ep_done:
                frame = self.env_wrapper.render()
                if play_params.RECORD_DIR is not None:
                    filepath = play_params.RECORD_DIR + '/Ep%03d_Step%04d.jpg' % (ep, step)
                    cv2.imwrite(filepath, frame)
                action = self.sess.run(self.actor_net.output, {self.state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                state, _, terminal = self.env_wrapper.step(action)
                state = self.env_wrapper.normalise_state(state)
                
                step += 1
                 
                # Episode can finish either by reaching terminal state or max episode steps
                if terminal or step == play_params.MAX_EP_LENGTH:
                    ep_done = True   
                    
        # Convert saved frames to gif
        if play_params.RECORD_DIR is not None:
            images = []
            for file in sorted(os.listdir(play_params.RECORD_DIR)):
                # Load image
                filename = play_params.RECORD_DIR + '/' + file
                if filename.split('.')[-1] == 'jpg':
                    im = cv2.imread(filename)
                    images.append(im)
                    # Delete static image once loaded
                    os.remove(filename)
                 
            # Save as gif
            imageio.mimsave(play_params.RECORD_DIR + '/%s.gif' % play_params.ENV, images, duration=0.01)  
                    
        self.env_wrapper.close()                   
    
