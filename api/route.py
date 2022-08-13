
from concurrent.futures.thread import ThreadPoolExecutor
import asyncio
from pydantic import BaseModel
from typing import List
from utils_ import format_result

from api.response import APIResponse

class Item(BaseModel):
    student_id: str
    subject: str
    program_level:int
    # history_topic: list
    masteries: dict
    history_score: list


def KyonRL_setup(app, RL_model):
    executor = ThreadPoolExecutor()

    def execute_api(item: Item):
        try:
            student_ID, action = RL_model.get_learningPoint(item.student_id, item.subject, str(item.program_level), item.masteries, item.history_score)
        except OSError as e:
            print(e)
            action = None

        return APIResponse.json_format(action)


    @app.post('/recommender')
    async def get_learningPoint(item: Item):

        return await asyncio.get_event_loop().run_in_executor(executor, execute_api, item)

        


