import logging
from fastapi import FastAPI  # FastAPI import
from pydantic import BaseModel
from clovax_rag_service import rag_service, initiate

# api 실행 방법
# #uvicorn web_main:app --reload

logger = logging.getLogger(__name__)
app = FastAPI()
initiate(is_console_mode=False)

class UserRequest(BaseModel):
    content: str

@app.post("/ai/api_recommender")
def api_recommender(user_request : UserRequest):
    logger.info(f"UserRequest: {user_request}")

    content = user_request.content
    rag_result = rag_service(content)
    return {"reply" : adjust_reply(rag_result)}

def adjust_reply(origin_reply):
    if origin_reply is not None:
        origin_reply = origin_reply.replace('https://***.****.**.**', 'https://www.data.go.kr')

    return origin_reply