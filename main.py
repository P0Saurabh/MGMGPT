from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Phi Model (Choose between Phi-1.5 or Phi-2)
model_name = "microsoft/phi-1_5"  # Use "microsoft/phi-2" for a more advanced model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create the text generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Request body schema
class ChatRequest(BaseModel):
    message: str

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "MGMGPT with Phi Model is running!"}

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    response = chatbot(user_message, max_length=100, do_sample=True, top_k=50)
    bot_reply = response[0]['generated_text']
    return {"reply": bot_reply}

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
