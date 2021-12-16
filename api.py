from fastapi import FastAPI
from model import load_model, process_sms
from pydantic import BaseModel

app = FastAPI()


class SMS(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/check/")
async def create_item(sms: SMS):
    model = load_model()  # load saved model
    txt = process_sms(sms.text)
    pred = (model.predict(txt) > 0.5).astype("int32").item()

    return pred == 1
