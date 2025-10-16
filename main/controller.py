import uvicorn
import json
import os
import shutil
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from model import *
from openai import Client
from security import create_client
from embeddings import get_device, load_model, load_tokenizer, generate_embeddings
from reasoning import text_qa
from blablador import blablador_get_models
import xml.etree.ElementTree as ET
import base64
import glob


app = FastAPI()
client: Client = create_client(type="openai")  # "openai" or "opensource"
client_opensource: Client = create_client(type="opensource")

def delete_files(filename_base_with_path):
    for f in glob.glob(filename_base_with_path + ".*"):
        os.remove(f)
        print(f"Deleted: {f}")

@app.get("/v1/blablador_models", status_code=200)
def get_blablador_modelnames() -> JSONResponse:

    models = blablador_get_models()
    
    return JSONResponse(
        status_code=200, 
        content=models
        )



path = os.environ["SEARCH_MODEL_PATH"]
ode_tree_path = os.environ["ODE_TREE_PATH"]

device = get_device()
model = load_model(model_path=path, device=device)
tokenizer = load_tokenizer(model_path=path)



@app.get("/v1/embeddings", status_code=200)
def get_embeddings(
    text: str
) -> JSONResponse:
    embeddings = generate_embeddings(
        text=text, 
        model=model, 
        tokenizer=tokenizer, 
        max_length=8192, 
        stride=4096, 
        device=device
    )
    
    return JSONResponse(
        status_code=200,
        content=embeddings
    )

@app.post("/v1/tqa", status_code=200)
def text_reasoning(
    request: TQARequest
) -> JSONResponse:
    response = TQAResponse(response=text_qa(request=request, client=client, client_opensource=client_opensource))

    return JSONResponse(
        status_code=200,
        content=response,
    )
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8200, timeout_keep_alive=180)
