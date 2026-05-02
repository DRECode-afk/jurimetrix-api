from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import pickle
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class Caso(BaseModel):
    texto: str

@app.post("/prever")
def prever(caso: Caso):
    X = vectorizer.transform([caso.texto])
    proba = modelo.predict_proba(X)[0]
    
    return {
        "chance_negar": float(proba[0]),
        "chance_conceder": float(proba[1])
    }