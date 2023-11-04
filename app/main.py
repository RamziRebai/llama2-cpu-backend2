import re
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware


base_dir=Path.cwd()

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

therapist_llm= Llama(model_path=f"{base_dir}/app/RamziRebai/llama-2-7b-therapist-v4/ggml-model-q5_k_m.gguf")

@app.get('/')
def home():
    return {"message": "Everything seems in order"}

@app.post('/execute-python/')
def generate(text: InputText):
    prompt_behavior=f"""<s>[INST] <<SYS>>Act like a Therapist advisor.You are excellent at identifying the mental health problem.You will provided with Patient request.The patient’s request is for assistance in dealing with the mental problem that you have dedicted.You must address this concern in a professional manner, while also being helpful to the patient as much as possible.Start with greeting the patient and calm him down.<</SYS>>

    Patient request: {text.text}
    Therapist response:
    [/INST]"""
    result=therapist_llm(prompt=prompt_behavior, max_tokens=512, temperature=0.7)['choices'][0]['text']
    cleaned_text = re.sub(r'<.*?>', '', result)
    cleaned_text= cleaned_text.replace("\n", '')
    cleaned_text= cleaned_text.strip()
    return {"stdout": cleaned_text}