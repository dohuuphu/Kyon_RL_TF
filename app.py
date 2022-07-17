import uvicorn

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from api.route import KyonRL_setup
from inference import Recommend_core

def set_up_app():
    app = FastAPI(name='kyonAI')
    origins = ["*"]
    # app.mount("/static/output", StaticFiles(directory="static/output", html = True), name="output")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


if __name__ == '__main__':
    

    app = set_up_app()

    model = Recommend_core()

    KyonRL_setup(app, model)
    
    uvicorn.run(app ,host='0.0.0.0', port=35010)