import math

import numpy as np

from variables import *


def sigmoid(x):
      return 1 / (1 + math.exp(-x))

# uniform random policy
def random_policy(q_values):
  return np.random.choice(len(q_values))

# greedy policy
def greedy(q_values):
  return np.argmax(q_values)

# epsilon greedy policy
def epsilon_greedy(q_values, epsilon):
  if epsilon < np.random.random():
    return np.argmax(q_values)
  else:
    return np.random.choice(len(q_values))


def norm_list(l):
    s = sum(l)
    new_l = [i/s for i in l]
    return new_l

def question_gen(skills, skill_prob):
  # Randomly selects 3-4 skills which represent the knowledge needed to answer a question
  # Input: list of all skill indices
  # Returns a list of skill indices
  # num_skills = random.randint(3,4)
  num_skills = 1
  skill_is = np.random.choice(skills, num_skills, p=skill_prob, replace=False)
  return skill_is

def test_gen(skills,num_q):
  # A test consists of num_q questions, randomly generated
  # All skills are involved
  # Returns a list of questions
  test = []
  skill_count = [0]*len(SKILL_LEVELS)
  skill_prob = norm_list(SKILL_LEVELS)
  for i in range(num_q):
    q = question_gen(skills, skill_prob)
    for skill in q:
      skill_count[skill] += 1
    test.append(q)
    skill_dist = norm_list(skill_count)
    new_skill_prob = []
    for si, v in enumerate(skill_prob):
      new_skill_prob.append(v-skill_dist[si])
    skill_prob = norm_list(skill_prob)

  return test


def test_gen_after(skills,num_q,action):
  # A test consists of num_q questions, randomly generated
  # All skills are involved
  # Returns a list of questions
  test = []
  skill_count = [0]*len(SKILL_LEVELS)
  skill_prob = norm_list(SKILL_LEVELS)
  for i in range(num_q):
    if i <2:
      test.append(action)
    else:
      q = question_gen(skills, skill_prob)
      for skill in q:
        skill_count[skill] += 1
      test.append(q)
      skill_dist = norm_list(skill_count)
      new_skill_prob = []
      for si, v in enumerate(skill_prob):
        new_skill_prob.append(v-skill_dist[si])
      skill_prob = norm_list(skill_prob)

  return test

def get_LPvalue(masteries:list) -> dict:
    list_LPvalue = {}
    for topic in LP_SEGMENT: 
      start_idx, stop_idx = LP_SEGMENT[topic]
      list_LPvalue.update({topic: {LP_VALUE_STR:[masteries[i] for i in range(start_idx, stop_idx, 1)],
                                  LP_DIFFICULT_STR:[LP_DIFFICULT_VALUE[i] for i in range(start_idx, stop_idx, 1)]}})

    return list_LPvalue
      
def calculate_topicWeight(masteries:list) -> dict:
  '''Calculate topic_weight by LP_value and LP_difficult
    Return format: {topic : weight (0-1)} '''
  dict_LPvalue = get_LPvalue(masteries)
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

def find_minTopicWeight(topic_weights:dict) -> str:
  '''find minimum topicWeight
    Return topic name'''
  return min(topic_weights, key=topic_weights.get)
  
def topic_recommender(curr_topic, masteries:list):

  topic_Weights = calculate_topicWeight(masteries)
  if curr_topic is None or topic_Weights[curr_topic] == 1:
    curr_topic = find_minTopicWeight(topic_Weights)

  return curr_topic

def mask_others_lp_not_in_topic(masteries:list ,topic:str):
  
  mask_masteries = LP_PER_TOPICS.copy()
  for idx, val in enumerate(masteries[LP_SEGMENT[topic][0]:LP_SEGMENT[topic][1]]):
    mask_masteries[idx] *= val

  return format_observation(mask_masteries)

def format_observation(list_ : list):
  return np.array([i*1.0 for i in list_], dtype=np.float64)