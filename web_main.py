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
    return {"reply" : rag_service(content)}
