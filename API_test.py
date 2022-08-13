# importing the requests library
import requests
import numpy as np
import json
# api-endpoint
URL = "http://0.0.0.0:35616/recommender"

# masteries = [1.0* i for i in np.random.randint(2, size=1034)]
with open('masteries.json','r') as f:
    masteries =  json.load(f)
print(masteries)

PARAMS = {
        "student_id": 1,
        "subject": "English",
        # "history_topic":["Verbs and objects"],
        # "history_action": [607],
        "history_action":[],
        "masteries": masteries,
        "history_score": [5]
    }

# sending get request and saving the response as response object
r = requests.post(url = URL, json = PARAMS)

# extracting data in json format
data = r.json()
print(data)


# extracting latitude, longitude and formatted address
# # of the first matching location
# latitude = data['results'][0]['geometry']['location']['lat']
# longitude = data['results'][0]['geometry']['location']['lng']
# formatted_address = data['results'][0]['formatted_address']

# # printing the output
# print("Latitude:%s\nLongitude:%s\nFormatted Address:%s"
# 	%(latitude, longitude,formatted_address))
