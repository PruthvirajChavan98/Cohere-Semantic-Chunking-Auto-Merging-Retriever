from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.core.settings import Settings
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.postprocessor.cohere_rerank import CohereRerank

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse, PlainTextResponse
import os

from Utils.sm_am_index_utils import SemanticAutoMergingIndexManager
from Utils.langchain_utils import get_chain
from Utils.CohereSettingsManager import SettingsManager

"""
FastAPI application to provide a query interface for a semantic search index.

This application allows users to query a semantic search index, leveraging 
Cohere's Large Language Models for response generation and reranking. 
It also provides endpoints for updating settings, loading or building the index, and streaming responses.
"""

class QueryData(BaseModel):
    """
    A data model to represent a user query.

    Attributes:
        query (str): The user's search query.
    """
    query: str

class CohereSettings(BaseModel):
    """
    A data model to represent Cohere API settings.

    Attributes:
        api_key (str): The API key for accessing Cohere's LLMs.
        model (str): The name or ID of the Cohere LLM to use.
    """
    api_key: str
    model: str

app = FastAPI()

settings_manager = SettingsManager()

embediing_model_folder_name = './bge_large_onnx'
model_name = "BAAI/bge-large-en-v1.5"

# Check if the model directory exists
if os.path.exists(embediing_model_folder_name):
    # Load the model from the existing directory
    embed_model = OptimumEmbedding(folder_name=embediing_model_folder_name)
    print("Model loaded from existing folder.")
else:
    # Download the model and save it to the folder
    print("Model not found locally. Downloading and saving.")
    OptimumEmbedding.create_and_save_optimum_model(model_name, embediing_model_folder_name)
    # After saving, load the model
    embed_model = OptimumEmbedding(folder_name=embediing_model_folder_name)

cohere_api_key = None
cohere_model = None
Settings.llm=None
Settings.embed_model=embed_model
sm_am_index_manager = None
index = None
retriever, query_engine = None, None
cohere_llm = None
chain = None


@app.post("/response", response_class=PlainTextResponse)
async def get_response(data: QueryData):
    """
    Endpoint to get a text response for a given query.

    Args:
        data (QueryData): A data model containing the user's query.

    Returns:
        str: The generated text response.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        chain = settings_manager.get_chain()
        cohere_rerank = settings_manager.get_cohere_rerank()
        nodes = retriever.retrieve(data.query)
        post_processed_nodes = cohere_rerank.postprocess_nodes(nodes=nodes, query_bundle=QueryBundle(data.query))
        text = "".join(node.text for node in post_processed_nodes)
        response = chain.invoke({"text": text, "topic": data.query})
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
async def stream_response(data: QueryData):
    """
    Endpoint to stream a text response for a given query.

    Args:
        data (QueryData): A data model containing the user's query.

    Returns:
        StreamingResponse: A streaming response containing the generated text.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        chain = settings_manager.get_chain()
        cohere_rerank = settings_manager.get_cohere_rerank()
        nodes = retriever.retrieve(data.query)
        post_processed_nodes = cohere_rerank.postprocess_nodes(nodes=nodes, query_bundle=QueryBundle(data.query))
        text = "".join(node.text for node in post_processed_nodes)
        
        async def generator():
            async for chunk in chain.astream({"text": text, "topic": data.query}):
                yield chunk.encode('utf-8')
        
        return StreamingResponse(generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query", response_class=PlainTextResponse)

async def query_index(data: QueryData):
    """
    Endpoint to query the index directly.

    Args:
        data (QueryData): A data model containing the user's query.

    Returns:
        str: The result of the query as a string.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        response = query_engine.query(data.query)
        return str(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/settings")
def get_settings():
    """
    Endpoint to retrieve the current settings.

    Returns:
        dict: A dictionary containing the current settings.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        return {"settings": settings_manager.get_settings()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_settings_and_load_or_build_index")
async def update_settings_and_load_or_build_index(
    cohere_api_key: str = Form(...),
    cohere_model: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint to update settings, load or build the index, and initialize related objects.

    Args:
        cohere_data_api_key (str): The API key for Cohere.
        cohere_data_model (str): The name or ID of the Cohere LLM to use.
        file (UploadFile): The uploaded file containing the index data.

    Returns:
        dict: A response indicating the success of the operation.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    global sm_am_index_manager, index, retriever, query_engine
    try:
        # Determine directories based on file name
        base_input_dir = "./uploads"
        os.makedirs(base_input_dir, exist_ok=True)
        input_dir = os.path.join(base_input_dir, file.filename.split('.')[0])
        persist_dir = f"{input_dir}_index"

        # Ensure the directories exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(persist_dir, exist_ok=True)

        # Save the uploaded file to the specified input directory
        file_location = f"{input_dir}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # Update settings through a manager that handles validation and rollback if needed
        settings_manager.update_cohere_settings(cohere_api_key, cohere_model)
        cohere_rerank = CohereRerank(
            top_n=20, api_key=cohere_api_key
        )

        sm_am_index_manager = SemanticAutoMergingIndexManager(
            rerank=cohere_rerank,
            embed_model=embed_model
        )

        index = sm_am_index_manager.load_or_build_semantic_automerging_index(
            persist_dir=persist_dir,
            input_dir=input_dir)

        retriever, query_engine = sm_am_index_manager.get_retriever_and_query_engine(
            automerging_index=index, 
            similarity_top_k=20
        )

        return {"response": "Settings and index updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

