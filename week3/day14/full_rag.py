import os
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Groq API key is not found")

client = Groq(api_key=my_api_key)

groqmodel = "openai/gpt-oss-20b"

model = SentenceTransformer("all-MiniLM-L6-v2")


# Knowledge Base
documents = [
    "Employees receive 24 days of paid leave per year.",

    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",

    "Employees receive Rs 3000 per month for gym reimbursement.",

    "Employees can claim Rs 2000 per month for home internet.",

    "Employees have a 90 day notice period."
]


# Create embeddings for all documents
document_embeddings = model.encode(documents)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def retrieve(qembedding):

    scores = []

    for i, document_embedding in enumerate(document_embeddings):

        score = cosine_similarity(
            qembedding,
            document_embedding
        )

        scores.append((score, documents[i]))

    scores.sort(reverse=True)

    return scores[0]


def ask_llm(question, context):

    sys_prompt = f"""
Answer in one line only.
Answer only based on the given context.
Do not hallucinate.

Context: {context}
"""

    system_message = {
        "role": "system",
        "content": sys_prompt
    }

    message = {
        "role": "user",
        "content": question
    }

    messages = [system_message, message]

    response = client.chat.completions.create(
        model=groqmodel,
        messages=messages
    )

    answer = response.choices[0].message.content

    return answer


query = "How much vacation do I get?"

qembedding = model.encode(query)

score, context = retrieve(qembedding)

answer = ask_llm(query, context)

print(answer)