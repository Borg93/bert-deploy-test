from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from classifer.model import Model, get_model

app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    score: float



@app.post("/predict", response_model=SentimentResponse)  #, 
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    res = model.predict(request.text)
    sentiment, score = res[0]['sequence'], res[0]['score']
    return SentimentResponse( sentiment = sentiment, score = score)
    
