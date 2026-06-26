from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict, train_and_save

app = FastAPI(title="API Predicción Abandono Crédito")

class SolicitudCredito(BaseModel):
    features: list[float]

@app.on_event("startup")
def startup():
    train_and_save()

@app.get("/")
def health():
    return {"status": "ok", "modelo": "XGBoost abandono credito"}

@app.post("/predecir")
def predecir(solicitud: SolicitudCredito):
    resultado = predict(solicitud.features)
    return resultado