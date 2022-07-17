import json
from starlette.responses import Response
from fastapi import File, UploadFile,status,Form

import requests
import json
import pandas as pd

class APIResponse():
        def json_format(response=None):
                if response != None: 
                        success = True
                        status_code = status.HTTP_200_OK
                        content = 'success'

                        return  Response(content=json.dumps(
                                                {
                                                'meta':{
                                                        'success': success,
                                                        'msg':content
                                                        },
                                                
                                                'response': response
                                                },
                                                ),
                                status_code=status_code,
                                media_type="application/json"
                                )

                else:
                        success = False
                        status_code = status.HTTP_501_NOT_IMPLEMENTED

                        content = 'Failed'
                return  Response(content=json.dumps(
                                                {
                                                'meta':{
                                                        'success': success,
                                                        'msg':content
                                                        }
                                                },
                                                ),
                                status_code=status_code,
                                media_type="application/json"
                        )


