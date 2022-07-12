from variables import *
from utils_ import sigmoid, test_gen, test_gen_after, topic_recommender, mask_others_lp_not_in_topic

from gym import spaces
from collections import Counter



import random

import numpy as np

def log_INFO(message):
  # print(message)
  logger.info(message)
  
class SimStudent():
      # An observation of the environment is a list of skill mastery, each value [0,1] represents the student's learning progress of a skill
  def __init__(self, intelligence = 50, luck=50):
    self.observation_space = [0, 1, 35] # min max shape
    self.action_space = [0, 34, 1] # min max shape
    self.v_min = -150.0
    self.v_max = 15.0

    self.true_masteries = np.zeros(len(SKILL_INDS)).astype(np.float32)

    self.answer_rate1 = 1
    self.answer_rate2 = 0.0
    self.forget_rate = 0
    self.learn_prob = 1


    # Initial test result
    # test = test_gen(SKILL_INDS, NUM_QUESTIONS_PER_TEST)
    self.masteries = self.true_masteries#self.test_update_masteries(test)
    # score = self.get_test_score(test, self.true_masteries)
    self.last_score = 0

    # Initialize history
    self.history = []
    self.history_topic = list()

  #===============
  def get_state_dims(self):
    return (self.observation_space[2],)

  def get_state_bounds(self):
    return np.array([self.observation_space[0]], dtype=np.float32), np.array([self.observation_space[1]], dtype=np.float32)

  def get_action_dims(self):
    return (self.action_space[2],)

  def get_action_bounds(self):  
     return np.array([self.action_space[0]], dtype=np.float32), np.array([self.action_space[1]], dtype=np.float32)
  
  def normalise_state(self, state):
    return state

  def normalise_reward(self, reward):
    return reward

  # def set_random_seed(self, seed)
  #   self._np_random, seed = seeding.np_random(seed)
  #       return [seed]

#=================================
  def mask_others_lp_not_in_topic(self,topic:str):
    
    mask_masteries = LP_PER_TOPICS.copy()
    for idx, val in enumerate(self.true_masteries[LP_SEGMENT[topic][0]:LP_SEGMENT[topic][1]]):
      mask_masteries[idx] *= val

    return mask_masteries

  def get_LPvalue(self) -> dict:
    list_LPvalue = {}
    for topic in LP_SEGMENT: 
      start_idx, stop_idx = LP_SEGMENT[topic]
      list_LPvalue.update({topic: {LP_VALUE_STR:[self.true_masteries[i] for i in range(start_idx, stop_idx, 1)],
                                  LP_DIFFICULT_STR:[LP_DIFFICULT_VALUE[i] for i in range(start_idx, stop_idx, 1)]}})

    return list_LPvalue

  def calculate_topicWeight(self) -> dict:
    '''Calculate topic_weight by LP_value and LP_difficult
      Return format: {topic : weight (0-1)} '''
    dict_LPvalue = self.get_LPvalue()
    topic_weights = {}
    for topic in dict_LPvalue:
      weight = 0
      total_element = 0
      for value, difficult in zip(dict_LPvalue[topic][LP_VALUE_STR], dict_LPvalue[topic][LP_DIFFICULT_STR]):
        weight += (value*difficult)
        total_element += difficult
      
      #update dict_result
      topic_weights.update({topic : (weight/total_element)})

    return topic_weights

  def find_minTopicWeight(self, topic_weights:dict) -> str:
    '''find minimum topicWeight
      Return topic name'''
    return min(topic_weights, key=topic_weights.get)
    
  def topic_recommender(self)-> str: 
    try:
      curr_topic = self.history_topic[-1] # define after
    except:
      curr_topic = None
    topic_Weights = self.calculate_topicWeight()
    if curr_topic is None or topic_Weights[curr_topic] == 1:
      curr_topic = self.find_minTopicWeight(topic_Weights)
      self.history = [] # reset history
    self.history_topic.append(curr_topic)
    return curr_topic

  def answer_question(self, question, masteries):
    raw_prob = 1
    for req in question:
      raw_prob = raw_prob*masteries[req]/SKILL_LEVELS[req]
    answer_prob = raw_prob*self.answer_rate1 + self.answer_rate2
    if answer_prob>1: answer_prob = 1
    if answer_prob<0: answer_prob = 0
    answer = np.random.choice([1,0],p=[answer_prob, 1-answer_prob])
    return answer

  def get_test_score(self, test, masteries):
    score = 0
    for question in test:
      answer = self.answer_question(question, masteries)
      if answer==1: score+=1
    return int(score/NUM_QUESTIONS_PER_TEST*10)

  def test_update_masteries(self, test):
    correct, all = {}, {}
    new_masteries = np.zeros(len(SKILL_INDS))
    for skill in SKILL_INDS:
      correct[skill] = 0
      all[skill] = 0
    for question in test:
      answer = self.answer_question(question, self.true_masteries)
      for skill in question:
        if answer==1: correct[skill]+=1
        all[skill]+=1
    for skill in all:
      if all[skill]!=0:
        new_masteries[skill] = int(correct[skill]/all[skill]*SKILL_LEVELS[skill])
    return [int(i) for i in new_masteries]
    
  def lesson_update_masteries(self, lesson_ind):
    new_masteries = self.true_masteries
    contained_skills = LESSONS[int(lesson_ind)]['skills']
    for skill,mastery in enumerate(self.true_masteries):
      if mastery<SKILL_LEVELS[skill] and skill in contained_skills:
        gain = np.random.choice([1,0],p=[self.learn_prob, 1-self.learn_prob])
        new_masteries[skill] += gain
    return [int(i) for i in new_masteries]

  def forget_update_masteries(self):
    new_masteries = self.true_masteries
    num_actions = len(self.history)
    for skill,mastery in enumerate(self.true_masteries):
      forget_prob = self.forget_rate*sigmoid(num_actions)*0.2
      loss = np.random.choice([1,0],p=[forget_prob, 1-forget_prob])
      if self.true_masteries[skill]>0: new_masteries[skill] -= loss
    return [int(i) for i in new_masteries]

  def is_complete_topic(self, topic_name):
    dict_LPvalue = self.get_LPvalue()
    for value in dict_LPvalue[topic_name][LP_VALUE_STR]:
      if value < 1 :
        return False
    return True    

  def step(self, action):   

    action = action.astype(np.int32)
    action_mapping = LP_SEGMENT[self.history_topic[-1]][0] + action
    # check_mastered = True
    # for i,m in enumerate(self.true_masteries):
    #   if m<SKILL_LEVELS[i]: check_mastered=False

    reward = 0
    log_INFO(f'action_before_filter: {action}')
    # reward for predict prev_action
    if len(self.history)>0:
      for i in range(len(self.history)-1 , -1 , -1):
        if self.history[i]==action_mapping: 
          reward+=-20
        else:
          break

    self.history.append(action_mapping) 
    if action >= (LP_SEGMENT[self.history_topic[-1]][1]-LP_SEGMENT[self.history_topic[-1]][0]) or action < 0:
      reward += -20
      num_same_act = self.count_consecutive_actions(action_mapping)
      # reward += (num_same_act-1)*(-1)

    else:
      if self.true_masteries[int(action_mapping)] == 1:
        reward += -20
      else:
        reward +=20
      log_INFO(f'action_mapping: {action_mapping}')
      if action_mapping in range(len(LESSONS)):
        self.true_masteries = self.lesson_update_masteries(action_mapping)
        # self.masteries = self.lesson_update_masteries(action)
        log_INFO(f'after lesson \| masteries: {Counter(self.masteries)} - true_m {Counter(self.true_masteries)}, reward {reward}')
        # reward += -1*penalty_weight # penalty term
      # if len(self.history)==MAX_NUM_ACTIONS or len(self.history)%NUM_ACTIONS_PER_TEST==0 or check_mastered==True:

        # Take a test after this action
        test = test_gen_after(SKILL_INDS,NUM_QUESTIONS_PER_TEST,action_mapping)
        # self.masteries = self.test_update_masteries(test)
        log_INFO(f'after test \| masteries: {Counter(self.masteries)} - true_m {Counter(self.true_masteries)}, reward {reward}')
        score = self.get_test_score(test, self.true_masteries)
        log_INFO(f'test score \| masteries: {Counter(self.masteries)} - true_m {Counter(self.true_masteries)}, reward {reward}')
        if score==0: reward+=0
        elif score==10: reward+=20 # max test score
        else: 
          reward += (score-self.last_score)*0.5# - 1*penalty_weight

        self.last_score = score


      num_same_act = self.count_consecutive_actions(action_mapping)
      # reward += (num_same_act-1)*(-1)

    # self.true_masteries = self.forget_update_masteries()

    # Check learing a topic is done
    done = self.is_complete_topic(self.history_topic[-1])

    # get and append new topic
    curr_topic = self.topic_recommender()
    segment_LPs = self.mask_others_lp_not_in_topic(curr_topic)

    # self.masteries = self.forget_update_masteries()
    log_INFO(f'Done step \| masteries: {Counter(self.masteries)} - true_m {Counter(self.true_masteries)}, reward {reward}')
    return self._get_obs(segment_LPs), np.array(reward, dtype=np.float32), done

  def reset(self):
    self.true_masteries = np.zeros(len(SKILL_INDS))
    for i,m in enumerate(self.true_masteries):
      self.true_masteries[i] = random.randint(0,SKILL_LEVELS[i])

    # # self.masteries = np.zeros(len(SKILL_INDS))
    # new_masteries = np.zeros(len(SKILL_INDS))
    # for i,m in enumerate(new_masteries):
    #   self.true_masteries[i] = random.randint(0,SKILL_LEVELS[i])
    # test = test_gen(SKILL_INDS,NUM_QUESTIONS_PER_TEST)
    # self.masteries = self.test_update_masteries(test)

    # score = self.get_test_score(test, self.true_masteries)
    self.last_score = 0
    self.history = []
    self.history_topic = []
    curr_topic = self.topic_recommender()
    segment_LPs = self.mask_others_lp_not_in_topic(curr_topic)
    return self._get_obs(segment_LPs)

  def _get_obs(self,masteries):
    return np.array([i*1.0 for i in masteries], np.float64)

  def preview(self):
    print('Learning probability:',self.learn_prob)
    print('Forget rate:',self.forget_rate)
    print('Known answer rate:',self.answer_rate1)
    print('Unknown answer rate:',self.answer_rate2)
    print('Skill masteries:',self.masteries)

  def count_consecutive_actions(self, action):
    count = 1
    if len(self.history)==0: return count
    else:
      # temp_acts = self.history
      # temp_acts.reverse()
      # for act in temp_acts:
      #   if act==action: count+=1
      #   else: break

      for action_ in self.history:
        if action_ == action: count+=1
      return count-1