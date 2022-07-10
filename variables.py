import pandas as pd
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
format = logging.Formatter('%(asctime)s - %(message)s')

handler = logging.FileHandler('logging_3_50.log')
handler.setFormatter(format)
logger.addHandler(handler)


# def prepare_env2(path ):

#     xl = pd.ExcelFile(path)

#     df1 = xl.parse(xl.sheet_names[3])

#     df2 = xl.parse(xl.sheet_names[4])

#     df22 = df2.loc[df2['Levels (Lớp)'] == 12]

#     lessons1 = df1['Lesson Contents (Nội dung)'].unique()
#     print(len(lessons1))
#     lessons2 = df22['Lesson Contents (Nội dung)'].unique()
#     print(len(lessons2))

#     lessons = list(lessons1) + list(lessons2)
#     len(lessons)

#     lessons.append('STOP')
#     lessons_r = []
#     skill_names_r = []
#     for _,row in df22.iterrows():
#         if row['Lesson Contents (Nội dung)']!="":
#             skill_cat = row['Categories (Mạch kiến thức)']
#             skill_diff = row['Difficulties (Độ khó)']
#             skill_name = str(row['Topics (Chủ đề)'])+'-'+str(row['Learning point (Chủ điểm)'])+'-'+str(row['Thứ tự LP'])+'-'+str(row['Difficulties (Độ khó)'])
#             skill = {'name':skill_name,'category':skill_cat,'difficulty':skill_diff}
#             skill_names_r.append(skill_name)
#             lesson = row['Lesson Contents (Nội dung)']
#             lesson_id = lesson
#             lessons_r.append({'id':lesson, 'skills':[skill_names_r.index(skill_name)]})
    
#     return lessons_r, skill_names_r

def prepare_env(path):

    xl = pd.ExcelFile(path)

    df1 = xl.parse(xl.sheet_names[3])

    df2 = xl.parse(xl.sheet_names[4])

    df22 = df2.loc[df2['Levels (Lớp)'] == 12]

    topics1 = df1['Topics (Chủ đề)'].unique()
    # print(topics1)
    topics2 = df22['Topics (Chủ đề)'].unique()
    # print(topics2)

    topics =   list(topics2) #list(topics1)
    
    difficult =  list(df22["Difficulties (Độ khó)"].values) #list(df1["Difficulties (Độ khó)"].values) +
    difficult = [0 if x != x else x for x in difficult]
    topics.append('STOP')

    learning_points = []
    map_lp_topics = dict()
    total_learning_points = 0

    # for index,topic in enumerate(topics1):
    #     map_lp_topics[topic] = len(df1.loc[df1['Topics (Chủ đề)'] == topic].values)
    for index,topic in enumerate(topics2):
        map_lp_topics[topic] = len(df22.loc[df22['Topics (Chủ đề)'] == topic].values)
    for key in map_lp_topics:
        start = total_learning_points
        total_learning_points += map_lp_topics[key]
        end= total_learning_points
        map_lp_topics[key]=[start,end]

    lessons_r = []
    skill_names_r = []
    # for _,row in df1.iterrows():
    #     if row['Lesson Contents (Nội dung)']!="":
    #         skill_cat = row['Categories (Mạch kiến thức)']
    #         skill_diff = row['Difficulties (Độ khó)']
    #         skill_name = str(row['Topics (Chủ đề)'])+'-'+str(row['Learning point (Chủ điểm)'])+'-'+str(row['Thứ tự LP'])+'-'+str(row['Difficulties (Độ khó)'])
    #         skill = {'name':skill_name,'category':skill_cat,'difficulty':skill_diff}
    #         skill_names_r.append(skill_name)
    #         lesson = row['Lesson Contents (Nội dung)']
    #         lesson_id = lesson
    #         lessons_r.append({'id':lesson, 'skills':[skill_names_r.index(skill_name)]})
    for _,row in df22.iterrows():
        if row['Lesson Contents (Nội dung)']!="":
            skill_cat = row['Categories (Mạch kiến thức)']
            skill_diff = row['Difficulties (Độ khó)']
            skill_name = str(row['Topics (Chủ đề)'])+'-'+str(row['Learning point (Chủ điểm)'])+'-'+str(row['Thứ tự LP'])+'-'+str(row['Difficulties (Độ khó)'])
            skill = {'name':skill_name,'category':skill_cat,'difficulty':skill_diff}
            skill_names_r.append(skill_name)
            lesson = row['Lesson Contents (Nội dung)']
            lesson_id = lesson
            lessons_r.append({'id':lesson, 'skills':[skill_names_r.index(skill_name)]})
    
    return map_lp_topics, difficult, lessons_r, skill_names_r

EXCEL_PATH = '/mnt/c/Users/dohuu/Desktop/kyons_AI/D4PG_kyon/Content Input English (Sample) - Kyons.xlsx'

LP_SEGMENT, LP_DIFFICULT_VALUE, LESSONS, SKILL_NAMES =  prepare_env(EXCEL_PATH)
# Number of questions per test
NUM_QUESTIONS_PER_TEST = 100
# Maximum number of times the student learns from the system, the course is terminated after this many lessons regardless of student's state
MAX_NUM_ACTIONS = 500
# The student takes a test every X actions
NUM_ACTIONS_PER_TEST = 1

penalty_weight = 1 # Modifier for penalty term in reward function

SKILL_INDS = range(len(SKILL_NAMES))
# Skill level: number of times an average student needs to learn to master a skill
SKILL_LEVELS = [1]*len(SKILL_NAMES)
# SKILL_SPACES = [i+1 for i in SKILL_LEVELS]
# SKILL_SPACES = [i+1 for i in LP_PER_TOPICS]
# Lessons: a lesson contains certain skills, learning a lesson may grant the student its skills
# LESSONS = [{'id':'01','skills':[0,1,2]},{'id':'02','skills':[1,3]},{'id':'03','skills':[0,1,4]},
#            {'id':'04','skills':[0,1,5]},{'id':'05','skills':[1,6]},{'id':'06','skills':[0,1,7]},
#            {'id':'07','skills':[0,1,8]},{'id':'08','skills':[1,9]},{'id':'09','skills':[0,1,10]},
#            {'id':'10','skills':[1,2,3,4]},{'id':'11','skills':[1,5,6,7]},{'id':'12','skills':[0,8,9,10]},
#            {'id':'13','skills':[0,1,11]},{'id':'14','skills':[1,2,5,8]},{'id':'15','skills':[1,3,6,9]},
#            {'id':'16','skills':[1,4,7,10]},{'id':'17','skills':[0,3,11]},{'id':'18','skills':[0,4,11]},
#            {'id':'19','skills':[1,5,6,11]},{'id':'20','skills':[0,10,11]}]




ACTIONS = ['LESSON '+l['id'] for l in LESSONS]
ACTIONS.append('STOP')


LP_PER_TOPICS = [1]*35
SKILL_SPACES = [i+1 for i in LP_PER_TOPICS]
NEW_SKILL_SPACES = [2.0]*35


MODEL_PATH = "./save/demo-eng12-grammar/" 

MODEL_CRITIC_PATH = ''
MODEL_POLICY_PATH = ''

LP_VALUE_STR = 'LP_value'
LP_DIFFICULT_STR = 'LP_difficut'
