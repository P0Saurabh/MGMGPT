from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow frontend to access backend (CORS policy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face model (Open-source model for zero-cost setup)
chatbot = pipeline("text-generation", model="gpt2")

# Request body schema
class ChatRequest(BaseModel):
    message: str

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "College ChatGPT Backend is running!"}

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    response = chatbot(user_message, max_length=50, num_return_sequences=1)
    bot_reply = response[0]['generated_text']
    return {"reply": bot_reply}

# Run with: uvicorn main:app --reload

