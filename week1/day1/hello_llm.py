import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key : raise ValueError("Groq API key is not found")
 
client = Groq(api_key = my_api_key)
model = "llama-3.3-70b-versatile"

messages = [
   {
     "role" : "user",
     "content" : "Give brief description of yourself"
   }
]

 #LLM CALL
response = client.chat.completions.create(
  model = model,
  messages = messages
 )

print(response.choices[0].message.content)
