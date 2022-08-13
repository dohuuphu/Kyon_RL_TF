import asyncio
import json
import time
from typing import Dict, Any, List, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import repeat
from aiohttp import ClientSession
import numpy as np

def http_get_with_requests(url: str, student_id: int, headers: Dict = {}, proxies: Dict = {}, timeout: int = 10) -> (int, Dict[str, Any], bytes):

    with open('masteries.json','r') as f:
        masteries =  json.load(f)

    PARAMS = {
            "student_id": student_id,
            "subject": "English",
            # "history_topic":["Verbs and objects"],
            # "history_action": [607],
            "history_action":[607],
            "masteries": masteries,
            "history_score": [5]
        }

    # sending get request and saving the response as response object
    response = requests.post(url = url, json = PARAMS)
    response_json = None
    try:
        response_json = response.json()
    except:
        pass

    response_content = None
    try:
        response_content = response.content
    except:
        pass

    return (response.status_code, response_json, response_content)


def http_get_with_requests_parallel(list_of_urls: List[str],list_student: List[int], headers: Dict = {}, proxies: Dict = {}, timeout: int = 10) -> (List[Tuple[int, Dict[str, Any], bytes]], float):
    t1 = time.time()
    results = []
    for i in range(100):
        executor = ThreadPoolExecutor(max_workers=3)
        for result in executor.map(http_get_with_requests, list_of_urls, list_student, repeat(headers), repeat(proxies), repeat(timeout)):
            results.append(result)
    t2 = time.time()
    t = t2 - t1
    return results, t


async def http_get_with_aiohttp(session: ClientSession, url: str, headers: Dict = {}, proxy: str = None, timeout: int = 10) -> (int, Dict[str, Any], bytes):
    # masteries = [1.0* i for i in np.random.randint(2, size=1034)]

    with open('masteries.json','r') as f:
        masteries =  json.load(f)

    PARAMS = {
            "student_id": 1,
            "subject": "English",
            # "history_topic":["Verbs and objects"],
            # "history_action": [607],
            "history_action":[607],
            "masteries": masteries,
            "history_score": [5]
        }



    response = await session.post(url=url, json=PARAMS)

    response_json = None
    try:
        response_json = await response.json(content_type=None)
        print(response_json)
    except json.decoder.JSONDecodeError as e:
        pass

    response_content = None
    try:
        response_content = await response.read()
    except:
        pass

    return (response.status, response_json, response_content)


async def http_get_with_aiohttp_parallel(session: ClientSession, list_of_urls: List[str], headers: Dict = {}, proxy: str = None, timeout: int = 10) -> (List[Tuple[int, Dict[str, Any], bytes]], float):
    t1 = time.time()
    results = await asyncio.gather(*[http_get_with_aiohttp(session, url, headers, proxy, timeout) for url in list_of_urls])
    t2 = time.time()
    t = t2 - t1
    return results, t


async def main():
    print('--------------------')

    # URL list
    urls = ["http://0.0.0.0:35616/recommender" for i in range(0, 3)]
    list_student = [1,2,3]

    # Benchmark aiohttp
    # session = ClientSession()
    # speeds_aiohttp = []
    # for i in range(0, 10):
    #     results, t = await http_get_with_aiohttp_parallel(session, urls)
    #     v = len(urls) / t
    #     print('AIOHTTP: Took ' + str(round(t, 2)) + ' s, with speed of ' + str(round(v, 2)) + ' r/s')
    #     speeds_aiohttp.append(v)
    # await session.close()

    print('--------------------')

    # Benchmark requests
    speeds_requests = []
    for i in range(0, 1):
        results, t = http_get_with_requests_parallel(urls, list_student=list_student)
        v = len(urls) / t
        print('REQUESTS: Took ' + str(round(t, 2)) + ' s, with speed of ' + str(round(v, 2)) + ' r/s')
        speeds_requests.append(v)

    # Calculate averages
    # avg_speed_aiohttp = sum(speeds_aiohttp) / len(speeds_aiohttp)
    avg_speed_requests = sum(speeds_requests) / len(speeds_requests)
    print('--------------------')
    # print('AVG SPEED AIOHTTP: ' + str(round(avg_speed_aiohttp, 2)) + ' r/s')
    print('AVG SPEED REQUESTS: ' + str(round(avg_speed_requests, 2)) + ' r/s')


asyncio.run(main())