# -*- coding: utf-8 -*-
# @Author  : zhousf
# @Function:
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import Response
import orjson
from uvicorn.config import Config
from uvicorn import Server


class Handler(object):
    def __init__(self):
        print("init handler")

    def run(self, msg):
        print("handler running...")
        return "handler done: {}".format(msg)


ML_MODELS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ML_MODELS["answer_to_everything"] = Handler()
    yield
    ML_MODELS.clear()

app = FastAPI(lifespan=lifespan)


def send_email(msg: str):
    print("send_email")
    result = ML_MODELS["answer_to_everything"].run(msg)
    time.sleep(5)
    print(result)


@app.get("/user/{msg}")
async def order(msg: str, task: BackgroundTasks):
    task.add_task(send_email, msg=msg)
    return Response(orjson.dumps({"message": "ok"}))


if __name__ == "__main__":
    config = Config(app=app, host="127.0.0.1", port=8010,  reload=False, debug=True, workers=1, limit_concurrency=2)
    server = Server(config)
    server.run()
