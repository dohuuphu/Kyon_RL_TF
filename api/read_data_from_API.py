import requests
import json
import pandas as pd

class GetReturnForBackend():
    def __init__(self) -> None:
        self.url = "https://api.tuhoconline.org/ai/lessons"
        self.headers = {"X-Authenticated-User":"kyons-ai-api-key"}
        self.respone = requests.request("GET",  self.url, headers=self.headers)
    def normalize_input(self):
        response_json = json.loads(self.respone.text)
        df = pd.DataFrame(response_json)
        print(df.shape)
        df = df[df['content'].notnull()]
        df = df[df['content']!=''] 
        topics = list(df['topic'].unique())
        

        map_lp_topics = dict()
        total_learning_points = 0

        for index,topic in enumerate(topics):
            map_lp_topics[topic] = len(df.loc[df['topic'] == topic].values)

        for key in map_lp_topics:
            start = total_learning_points
            total_learning_points += map_lp_topics[key]
            end= total_learning_points
            map_lp_topics[key]=[start,end]
        lessons_r = []
        skill_names_r = []
        list_row = []
        for _,row in df.iterrows():
            if row['content']:
                skill_cat = row['category']
                skill_name = str(row['id'])+ '-' + str(row['topic'])+'-'+str(row['priority'])
                skill = {'name':skill_name,'category':skill_cat}
                skill_names_r.append(skill_name)
                lesson = row['content']
                lesson_id = lesson
                lessons_r.append({'id':lesson, 'skills':[skill_names_r.index(skill_name)]})
                list_row.append(row)

        return map_lp_topics, lessons_r, skill_names_r , list_row
        
    def return_result (self,topic_name:str,id:int):
        lp_segment, lessons, skill_names, list_row = self.normalize_input()
        return list(filter(lambda topic: topic['topic'] == topic_name, list_row))[id]
        
if __name__ == "__main__":
    backend = GetReturnForBackend()
    lesson = backend.return_result('Unit 1',0)
    print(lesson)


