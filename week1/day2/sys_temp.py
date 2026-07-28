import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key : raise ValueError("Groq API key is not found")
 
client = Groq(api_key = my_api_key)
model = "llama-3.3-70b-versatile"

message_system = {
        "role" : "system",
        "content" : "you are a brand manager who suggest name for my food company name should be in one word suggest only word"
    }


message = {
     "role" : "user",
     "content" : "suggest me a name for my food company"
   }


 #LLM CALL
messages = [message_system , message]
response = client.chat.completions.create(
  model = model,
  messages = messages,
  temperature = 2
 )

print(response.choices[0].message.content)
