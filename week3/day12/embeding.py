import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
model = SentenceTransformer("all-MiniLM-L6-v2") #384 vectors

t1 = "there are 24 paid leaves"
t2 = "cat is a wild animal"

v1 = model.encode(t1)
v2 = model.encode(t2)

print(cosine_similarity(v1,v2))