import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key : raise ValueError("Groq API key is not found")
 
client = Groq(api_key = my_api_key)
model = "llama-3.3-70b-versatile"

def llm_ans(prompt):
    message = {
        "role" : "user",
        "content" : prompt
    }
    messages = [message]
    response = client.chat.completions.create(model = model , messages = messages)
    ans = response.choices[0].message.content
    return ans

bad_prompt = """ 
#ROLE
you are a support assistant at a mobile/laptop company

#task
you have to classify the issue in a category

#CONSTRAINTS
you have to classify the issue in one of the three categories - Billing , Refund , Technical

#OUTPUT FORMED
your output should be in a one word only. The one word should be one of the categories given in constraints

#EXAMPLE
for instance if a user complains says he wants a refund then the category is refund

#FALL BACK
if the issue is unrelated to any of the category mentioned in constarints , then the ans should be OTHER

This is user complaint:
my laptop is not working 
"""
print(llm_ans(bad_prompt))