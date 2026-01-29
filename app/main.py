import os
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from utils import get_weather, get_mandi_price

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GOV_API_KEY = os.getenv("GOV_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
WEATHER_KEY = os.getenv("WEATHER_KEY")

client = genai.Client(api_key=GEMINI_KEY)

with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)


class FarmData(BaseModel):
    n: int
    p: int
    k: int
    ph: float
    rainfall: float
    city: str

class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def home():
    """Prevents 404 errors on the root URL"""
    return {"message": "AgroSmart Intelligence API is active. Visit /docs for technical details."}

@app.post("/predict")
async def predict_crop(data: FarmData):
    """Predicts the best crop and provides AI strategy"""
    weather = get_weather(data.city, WEATHER_KEY)
    
    if not weather:
        return {"error": "Weather data unavailable for the specified location."}

    features = np.array([[data.n, data.p, data.k, weather['temp'], weather['humidity'], data.ph, data.rainfall]])
    
    crop = model.predict(features)[0]
    
    price = get_mandi_price(crop, GOV_API_KEY)
    
    prompt = f"""
    Context: A professional farmer in {data.city} is growing {crop}. 
    Data: Temp {weather['temp']}°C, Humidity {weather['humidity']}%, Market Price {price}.
    Task: Provide a high-level 3-point biological and market strategy.
    """
    
    response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    
    return {
        "crop": crop,
        "price": price,
        "advice": response.text
    }

@app.post("/chat")
async def chat_with_agri_bot(request: ChatRequest):
    """Multilingual AI Agronomist Chatbot endpoint"""
    try:
        prompt = (
            "You are the AgroSmart Virtual Agronomist. "
            "Answer the farmer's question professionally and concisely. "
            "If the user asks in Hindi, respond in Hindi. If in English, respond in English. "
            f"Question: {request.message}"
        )
        
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        
        return {"reply": response.text}
    except Exception as e:
        print(f"Chat Error: {e}")
        return {"reply": "I am processing your request. Please try again in a few seconds."}
