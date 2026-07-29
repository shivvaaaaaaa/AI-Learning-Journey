import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key : raise ValueError("Groq API key is not found")
 
client = Groq(api_key = my_api_key)
model = "llama-3.3-70b-versatile"
role = "user"

from pydantic import BaseModel

class Ticket (BaseModel):
    name : str
    email : str
    issue : str

schema = Ticket.model_json_schema()
response_format = {
        "type" : "json_object"
    }

system_prompt = f''' Extract the personal information  and also issue from the ticket strictly based on schema and give a json output , {schema}'''

message_system = {
    "role" : "system", 
    "content" : system_prompt
}

text = "My name is shiva sharma and i have a iphone which is stop working and my email is abc@gmail.com , i am form delhi"

prompt = f'''this is customer ticket please extract the personal information from this , {text}'''

message = { 
    "role" : "user", 
    "content" : prompt
}

messages = [message_system , message]
response = client.chat.completions.create( model = model , messages = messages ,response_format = response_format )

answer = response.choices[0].message.content
print(answer)



