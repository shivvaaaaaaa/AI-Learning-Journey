import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key : raise ValueError("Groq API key is not found")
 
client = Groq(api_key = my_api_key)
model = "llama-3.3-70b-versatile"

#step1
knowledge_base={
    "age": "The age of shiva is 20"
    "net worth": "The networth of shiva is 2000"
}

#step2 retrieval
def retrieve_info(question):
    if "age" in question:
        return knowledge_base[age]
    elif "networth" in question:
        return knowledge_base[net worth]
    else:
        return None

def ask_llm(question):
    context = retrieve_info(question)

    sys_prompt = f"""answer in one line only. Answer only based on this context. do not hallucinate. context: {context}"""
    system_message = {
        "role": "system"
        "content": sys_prompt
    }

    message = {
        "role": "user"
        "content": question
    }

    messages = [message, system_message]
    response = client.chat.completions.create(model = model , messages = messages)
    answer = response.choices[0].message.content
    return answer

question = "what is the age of shiva?"
print(ask_llm(question))


