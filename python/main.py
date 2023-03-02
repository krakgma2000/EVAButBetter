from typing import Union
from fastapi import Request
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    isReceived: bool
    type: str
    content: str


@app.post("/get_msg")
async def get_msg(request: Request):
    data = await request.json()
    content = data["content"]
    content["isReceived"] = True
    return content

@app.get("/test")
async def test():
    return {"message": "what? do you want to ask to me"}

# if(data["request"] is None):
#     content["content"] = "Bad request"
#     content["type"] = "sysmsg"
#     content["isReceived"] = True
#     return content