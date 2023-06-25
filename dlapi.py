from fastapi import FastAPI # Bring in some light-weight wrappers for the DL API 
from pydantic import BaseModel # Let's use the pydantic library to define a data model for our API
import pickle # We'll use pickle to load the model
import pandas as pd # We'll use pandas to read the dataset

# Creating a class which describes the data model for our API
class _sentiment_analysis(BaseModel):
    text : str # The text of the review
  
# Let's load the model using pickle
model_path = 'sentiment_analysis_model.p'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# Creating a decorator to tell what the route is going to be
@app.post('/')
async def scoring_endpoint(item: _sentiment_analysis):

    # Let's send the text to the model and get the prediction
    sentiment = model.predict([item.text])

    # Let's return the prediction
    return {"Sentiment": "Positive" if sentiment == 1 else "Negative"}
