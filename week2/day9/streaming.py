import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key : raise ValueError("Groq API key is not found")
 
client = Groq(api_key = my_api_key)
model = "llama-3.3-70b-versatile"

prompt = "what is my name see old data of mine give information of me "
messages = [
   {
     "role" : "user",
     "content" : prompt
   }
]

#  #LLM CALL
# response = client.chat.completions.create(
#   model = model,
#   messages = messages
#  )

# print(response.choices[0].message.content)

response = client.chat.completions.create(model = model , messages = messages , stream = True)

#for print the ans 
for chunk in response:
    content =  chunk.choices[0].delta.content
    if content:
        print(content , end = "" , flush = True)