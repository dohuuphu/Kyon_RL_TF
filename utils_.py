import math
import numpy as np
import ast
import os
import glob
import shutil
from variables import *

from collections import deque, OrderedDict
import pandas as pd

from os.path import dirname, join, basename, exists

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

def get_LPvalue(masteries:list, level:str) -> dict:
    list_LPvalue = {}
    lp_segment = LESSON_DATABASE.get_LPsegments(level)
    # lp_difficult_value = LESSON_DATABASE.get_LP_difficult_value(level)
    try:
      for topic in lp_segment: 
        start_idx, stop_idx = lp_segment[topic]
        list_LPvalue.update({topic: {LP_VALUE_STR:[masteries[i] for i in range(start_idx, stop_idx, 1)]
                                    }}) #LP_DIFFICULT_STR:[lp_difficult_value[i] for i in range(start_idx, stop_idx, 1)]
    except:
      print(f'ERROR - get_LPvalue -topic {topic}  ')
    return list_LPvalue
      
def calculate_topicWeight(masteries:list, level:str) -> dict:
  '''Calculate topic_weight by LP_value and LP_difficult
    Return format: {topic : weight (0-1)} '''
  dict_LPvalue = get_LPvalue(masteries, level)
  topic_weights = {}
  for topic in dict_LPvalue:
    weight = 0
    for value in dict_LPvalue[topic][LP_VALUE_STR]:
      weight += (value)
    
    #update dict_result
    topic_weights.update({topic : (weight/len(dict_LPvalue[topic][LP_VALUE_STR]))})

  return topic_weights

def find_minTopicWeight(topic_weights:dict) -> str:
  '''find minimum topicWeight
    Return topic name'''
  return max(topic_weights, key=topic_weights.get)
  
def topic_recommender(masteries:list, level:str, curr_topic=None):

  topic_Weights = calculate_topicWeight(masteries, level)
  if curr_topic is None or topic_Weights[curr_topic] == 1:
    curr_topic = find_minTopicWeight(topic_Weights)

  return curr_topic

def mask_others_lp_not_in_topic(masteries:list , topic:str, level:str):
  Lp_segment = LESSON_DATABASE.get_LPsegments(level)
  mask_masteries = LP_PER_TOPICS.copy()
  for idx, val in enumerate(masteries[Lp_segment[topic][0]:Lp_segment[topic][1]]):
    mask_masteries[idx] *= val

  return format_observation(mask_masteries)

def format_observation(list_ : list):
  return np.array([i*1.0 for i in list_], dtype=np.float64)


def format_result( student_ID:int, id:int, topic_name:str, level:str):
  '''format result as dictionary to return backend'''

  return LESSON_DATABASE.get_data_parsed(level)[topic_name][int(id[0])]
  

def load_deque(id_user:str, level:str):
    exp_buffer = deque()
    try:
      with open(join(ROOT_DATABASE, EXP_BUFFER, f'{id_user}_{level}.txt'),'r') as f:
          infomation = f.readlines()
          for line in infomation:
              line = line.split('\t')
              state = np.array(ast.literal_eval(line[0]))
              action = np.array(ast.literal_eval(line[1]))
              reward = np.array(ast.literal_eval(line[2]))
              exp_buffer.append((state,action,reward))
    except:
      print(f'new user {id_user}')

    return exp_buffer
            
def save_deque(id_user, level, deque):
  with open(join(ROOT_DATABASE, EXP_BUFFER, f'{id_user}_{level}.txt'), 'w') as f:
    for i in range(len(list(deque))):
        state = list(deque[i])[0]
        action = list(deque[i])[1]
        reward = list(deque[i])[2]
        
        format_save = str(list(state)) + '\t' + str(action) + '\t' + str(reward) + '\n' 
        f.write(format_save)

def log_history(id, id_user, subject:str, level:str, masteries:list, recommend_action:int, action_state:int, score, dataframe:pd.DataFrame = None):
  user_name = f'{id_user}_{subject}_{level}'

  df = pd.DataFrame([[id, id_user, subject, level, masteries, recommend_action, action_state, score]], 
            columns=['id', 'user_id', 'subject', 'level', 'masteries', 'recommend_action', 'action_state', 'score'])

  if dataframe is None:
    df.to_csv(join('history',f'{user_name}.csv'),index=False)
  else:
    df.to_csv(join('history', f'{user_name}.csv'), mode='a', header=False, index=False)
    
  # if os.path.exists(join('history',f'{user_name}.csv')):
  #   df = pd.read_csv(join('history',f'{user_name}.csv'))
  #   df_new = pd.DataFrame([[id_user, subject, level, masteries, curr_topic, recommend_action, score]], columns=['user_id', 'subject', 'level', 'masteries', 'curr_topic', 'recommend_action', 'score'])
  #   df = pd.concat([df, df_new])
  #   df.to_csv(join('history', f'{user_name}.csv'), index=False)
  
  # else:
  #   df = pd.DataFrame([[id_user, subject, level, masteries, curr_topic, recommend_action, score]], columns=['user_id', 'subject', 'level', 'masteries', 'curr_topic', 'recommend_action', 'score'])
  #   df.to_csv(join('history',f'{user_name}.csv'),index=False)

  

def save_masteries(id_user:str, level:str, masteries:list):
  name_folder = f'{id_user}_{level}'
  folder_path = create_folder(name_folder)
  if len(get_totalFile_Masteries(id_user, level)) > 10:
    remove_item_inFolder(folder_path)
  name_txt = f'{len(get_totalFile_Masteries(name_folder))}.txt'
  with open(join(folder_path, name_txt), 'w') as f:
        f.write(str(masteries))

def read_masteries(id_user, idx= -1):
  dict_masteries = get_totalFile_Masteries(id_user)
  name_masteries = list(dict_masteries)[idx]

  info = open(dict_masteries[name_masteries], 'r').readline()
  masteries = list(info.replace('[','').replace(']','').split(","))

  return [float(i.strip()) for i in masteries]

def remove_item_inFolder(folder):
    for i in glob.glob(f'{folder}/*'):
        os.remove(i)


def create_folder( id_user, remain = True):
    ''' create folder with path, remove old folder if it exist'''
    folder_path = join(ROOT_DATABASE, MASTERIES_STORAGE, str(id_user))
    if exists(folder_path):
        try:
            if remain: return folder_path
            shutil.rmtree(folder_path)
        except OSError as e:
            pass

    os.makedirs(folder_path)
    
    return folder_path

def get_totalFile_Masteries(id_user, level, format = '.txt') -> dict:
    dict_txt = {}
    folder_path = join(ROOT_DATABASE, MASTERIES_STORAGE, str(id_user))
    for item in os.listdir(folder_path):
      dict_txt.update({int(item.replace('.txt', '')):join(folder_path,item)})
      
    dict_txt = OrderedDict(sorted(dict_txt.items()))

    return dict_txt


def init_database(remain = True):
    masteries_storage_path = join(ROOT_DATABASE, MASTERIES_STORAGE)
    expbuffer_storage_path = join(ROOT_DATABASE, EXP_BUFFER)
    create_database_folder(masteries_storage_path, remain)
    create_database_folder(expbuffer_storage_path, remain)
    return

def create_database_folder(folder_path, remain = True):
    if exists(folder_path):
        try:
            if remain: return 
            shutil.rmtree(folder_path)
        except OSError as e:
            pass
    
    if not exists(ROOT_DATABASE):
      os.makedirs(ROOT_DATABASE)
    
    os.makedirs(folder_path)
    
