from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class SentimentAnalysisInput(BaseModel):
    text: str

app = FastAPI()

model_path = 'sentiment_analysis_model.p'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.post('/predict')
async def scoring_endpoint(item: SentimentAnalysisInput):
    print(item)  # Print the received item for debugging
    sentiment = model.predict([item.text])
    return {"Sentiment": "Positive" if sentiment[0] == 1 else "Negative"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
