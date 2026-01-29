import os
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from typing import Dict
from utils import get_weather, get_mandi_price

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session tracker (In-memory)
user_sessions: Dict[str, Dict] = {}

GOV_API_KEY = os.getenv("GOV_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
WEATHER_KEY = os.getenv("WEATHER_KEY")

client = genai.Client(api_key=GEMINI_KEY)

# Use the stable model name to avoid the 404 error seen in logs
MODEL_NAME = "gemini-1.5-flash" 

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
    user_id: str  # Added to track which user is choosing which language

@app.get("/")
async def home():
    return {"message": "AgroSmart Intelligence API is active. Visit /docs for technical details."}

@app.post("/predict")
async def predict_crop(data: FarmData):
    weather = get_weather(data.city, WEATHER_KEY)
    if not weather:
        return {"error": "Weather data unavailable."}

    features = np.array([[data.n, data.p, data.k, weather['temp'], weather['humidity'], data.ph, data.rainfall]])
    crop = model.predict(features)[0]
    price = get_mandi_price(crop, GOV_API_KEY)
    
    prompt = f"Farmer in {data.city}, Crop: {crop}, Temp: {weather['temp']}C. Give a 3-point strategy."
    response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    
    return {"crop": crop, "price": price, "advice": response.text}
    
@app.post("/chat")
async def chat_with_agri_bot(request: ChatRequest):
    uid = request.user_id
    msg = request.message.strip().lower()

    # Handle Reset/Restart
    if msg == "reset" or msg == "init":
        if uid in user_sessions:
            del user_sessions[uid]
        user_sessions[uid] = {"step": "lang_selection", "lang": None}
        return {"reply": "Welcome back! Please select your language: \n1. English \n2. Hindi (हिंदी)"}

    if uid not in user_sessions:
        user_sessions[uid] = {"step": "lang_selection", "lang": None}
        return {"reply": "Please select your language: 1. English or 2. Hindi."}

    session = user_sessions[uid]

    # Handle Language Selection
    if session["step"] == "lang_selection":
        if "1" in msg or "english" in msg:
            session.update({"step": "chatting", "lang": "English"})
            return {"reply": "English set. I can help with soil, crops, weather, and Mandi prices. What is your question?"}
        elif "2" in msg or "hindi" in msg:
            session.update({"step": "chatting", "lang": "Hindi"})
            return {"reply": "हिंदी सेट की गई है। मैं मिट्टी, फसल और मंडी भाव में मदद कर सकता हूँ। आपका प्रश्न क्या है?"}
        return {"reply": "Choose 1 or 2."}

    # Strict Contextual Chatting
    try:
        lang = session["lang"]
        system_instruction = (
            f"You are the AgroSmart Virtual Assistant. Respond strictly in {lang}. "
            "IMPORTANT: Only answer questions related to agriculture, soil (NPK, pH), "
            "weather, and crop market prices as featured on the AgroSmart website. "
            "If a user asks about unrelated topics (movies, sports, etc.), politely "
            "decline and ask them to stick to farming queries."
        )
        
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=f"{system_instruction}\nUser: {request.message}"
        )
        return {"reply": response.text}
    except Exception as e:
        return {"reply": "Error connecting to AI. Please try again."}
