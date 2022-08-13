

import requests
import json
import pandas as pd

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
format = logging.Formatter('%(asctime)s - %(message)s')

handler = logging.FileHandler('logging_3_50.log')
handler.setFormatter(format)
logger.addHandler(handler)


class Lesson_Database():
    def __init__(self):
        self.respone = self.get_lessonFrombackend()
        (self.lp_segment, self.lessons, self.skill_names, 
            self.lp_difficult_value, self.data_parsed) = self.preprocess_data() 
        

    def get_lessonFrombackend(self)->requests:
        url = "https://api.tuhoconline.org/ai/lessons"
        headers = {"X-Authenticated-User":"kyons-ai-api-key"}
        return requests.request("GET",  url, headers=headers)

    def preprocess_data(self):
        response_json = json.loads(self.respone.text)
        df = pd.DataFrame(response_json)
        df = df[df['content'].notnull()]
        df = df[df['content']!=''] 
        topics = list(df['topic'].unique())
        topics = sorted(topics)

        map_lp_topics = dict()
        total_learning_points = 0
        masteries = dict()
        

        for index, topic in enumerate(topics):
            # if index < 3:
            map_lp_topics[topic] = len(df.loc[df['topic'] == topic].values)

        for key in map_lp_topics:
            masteries[key] = []
            start = total_learning_points
            total_learning_points += map_lp_topics[key]
            end= total_learning_points
            map_lp_topics[key]=[start,end]
        lessons_r = []
        skill_names_r = []
        list_row = []
        for _,row in df.iterrows():
            if row['content']:
                masteries[str(row['topic'])].append(row['id'])
                skill_cat = row['category']
                skill_name = str(row['id'])+ '-' + str(row['topic'])+'-'+str(row['priority'])
                skill = {'name':skill_name,'category':skill_cat}
                skill_names_r.append(skill_name)
                lesson = row['content']
                lesson_id = lesson
                lessons_r.append({'id':lesson, 'skills':[skill_names_r.index(skill_name)]})

        for key in masteries:
            masteries[key]= sorted(masteries[key])

        return map_lp_topics, lessons_r, skill_names_r, [1.0]*len(lessons_r), masteries
        
    # def return_result (self,topic_name:str,id:int):
    #     lp_segment, lessons, skill_names, masteries = self.normalize_input()
    #     return masteries[topic_name][id]

    def get_LPsegments(self, level:str):
        return self.lp_segment[level]
    
    def get_lessons(self, level:str):
        return self.lessons[level]
    
    def get_skill_names(self, level:str):
        return self.skill_names[level]

    def get_LP_difficult_value(self, level:str):
        return self.lp_segment[level]

    def get_data_parsed(self, level:str):
        return self.data_parsed[level]

    def get_skill_level(self, level:str):
        return [1]*len(self.get_skill_names[level])
        
    def get_skill_IND(self, level:str):
        return range(len(self.get_skill_names(level)))
        

LESSON_DATABASE = Lesson_Database()
# LP_SEGMENT, LESSONS, SKILL_NAMES, LP_DIFFICULT_VALUE, DATA_PARSED =  data_backend.normalize_input()

LP_PER_TOPICS = [1]*80
NUM_QUESTIONS_PER_TEST = 100


LP_VALUE_STR = 'LP_value'
LP_DIFFICULT_STR = 'LP_difficut'

#database
ROOT_DATABASE = './database'
EXP_BUFFER = 'exp_buffer'
MASTERIES_STORAGE = 'masteries'