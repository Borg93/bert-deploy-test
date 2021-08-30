from typing import Dict
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from classifer.model import Model, get_model

app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    score: float


@app.get("/")
async def root():
    return {"message": "hello world"}

@app.post("/predict", response_model=SentimentResponse)  #, 
async def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    res = model.predict(request.text)
    sentiment, score = res[0]['sequence'], res[0]['score']
    return SentimentResponse( sentiment = sentiment, score = score)

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


    
