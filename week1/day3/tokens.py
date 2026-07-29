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

prompt1 = "Hi!"
prompt2 = "Explain time travel in 1000 words"
prompt3 = "Write a essay on ML in 1000 words"

prompts = [prompt1,prompt2,prompt3]

for prompt in prompts:
    message = {
        "role" : role ,
        "content" : prompt
    }


    messages = [message]

    response = client.chat.completions.create(model = model , messages = messages , max_tokens = 5000)
    usage = response.usage

    print(f"Prompt:{prompt} -----> Your Tokens = {usage.prompt_tokens} , Completion Tokens:{usage.completion_tokens} , Total Tokens{usage.total_tokens} Finish Reason: {response.choices[0].finish_reason}")




