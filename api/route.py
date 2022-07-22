
from concurrent.futures.thread import ThreadPoolExecutor
import asyncio
from pydantic import BaseModel
from typing import List
from utils_ import format_result

from api.response import APIResponse

class Item(BaseModel):
    student_id: int
    subject: str
    history_topic: list
    history_action: list
    masteries: list
    history_score: list


def KyonRL_setup(app, RL_model):
    executor = ThreadPoolExecutor()

    def execute_api(item: Item):
        # item.masteries = [i*1.0 for i in item.masteries]
        try:
            student_ID, action, topic_name = RL_model.get_learningPoint(item.student_id, item.subject, item.history_topic, item.history_action, item.masteries, item.history_score)
            result = format_result(student_ID, action, topic_name)

        except OSError as e:
            print(e)
            result = None

        return APIResponse.json_format(result)


    @app.post('/recommender')
    async def get_learningPoint(item: Item):

        return await asyncio.get_event_loop().run_in_executor(executor, execute_api, item)

        


