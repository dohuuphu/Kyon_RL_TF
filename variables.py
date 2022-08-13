

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
        (self.lp_segment, self.data_parsed) = self.preprocess_data() 
        

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

        masteries_new = {"10":dict(), "11":dict(), "12":dict()}
        map_lp_topics_new = dict()

        for index, topic in enumerate(topics):
            list_lp_10 = []
            list_lp_11 = []
            list_lp_12 = []
            df_temp = df.loc[(df['topic'] == topic)].values
            for i in range(len(df_temp)):
                if 10 in df_temp[i][5]:
                    list_lp_10.append(df_temp[i][0])
                    masteries_new['10'][topic] = list_lp_10
                if 11 in df_temp[i][5]:
                    list_lp_11.append(df_temp[i][0])
                    masteries_new['11'][topic] = list_lp_11
                if 12 in df_temp[i][5]:
                    list_lp_12.append(df_temp[i][0])
                    masteries_new['12'][topic] = list_lp_12

        for key in masteries_new:
            total_learning_points = 0
            map_lp_topics_new[key] = {}
            for sub_key in masteries_new[key]:
                map_lp_topics_new[key][sub_key] = []
                start = total_learning_points
                total_learning_points += len(masteries_new[key][sub_key])
                end= total_learning_points
                map_lp_topics_new[key][sub_key]=[start,end]

        for key in masteries_new:
            for sub_key in masteries_new[key]:
                masteries_new[key][sub_key]= sorted(masteries_new[key][sub_key])

        return map_lp_topics_new,  masteries_new
        
    # def return_result (self,topic_name:str,id:int):
    #     lp_segment, lessons, skill_names, masteries = self.normalize_input()
    #     return masteries[topic_name][id]

    def get_LPsegments(self, level:str):
        return self.lp_segment[level]
    
    # def get_lessons(self, level:str):
    #     return self.lessons[level]
    
    # def get_skill_names(self, level:str):
    #     return self.skill_names[level]

    # def get_LP_difficult_value(self, level:str):
    #     return self.lp_segment[level]

    def get_data_parsed(self, level:str):
        return self.data_parsed[level]

    def get_skill_level(self, level:str):
        return [1]*self.get_len_masteries(level)

    def get_len_masteries(self, level:str):
        return list(self.lp_segment[level].items())[-1][1][-1]

    def get_skill_IND(self, level:str):
        return range(self.get_len_masteries(level))
        

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