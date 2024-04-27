from fastapi import HTTPException
from llama_index.llms.cohere import Cohere
from langchain_cohere import ChatCohere
from llama_index.postprocessor.cohere_rerank import CohereRerank

from Utils.langchain_utils import get_chain

class SettingsManager:
    def __init__(self):
        self.cohere_api_key = None
        self.cohere_model = None
        self.llm = None
        self.cohere_rerank = None
        self.chain = None

    def update_cohere_settings(self, api_key, model):
        self.cohere_api_key = api_key
        self.cohere_model = model
        try:
            self.llm = Cohere(model=model, api_key=api_key, temperature=0.0)
            self.cohere_rerank = CohereRerank(api_key=api_key)
            self.chain = get_chain(llm=ChatCohere(cohere_api_key=api_key, model=model, temperature=0.0))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

    def get_cohere_rerank(self):
        return self.cohere_rerank

    def get_chain(self):
        return self.chain
    
    def get_settings(self):
        return {"cohere_api_key": self.cohere_api_key, "cohere_model": self.cohere_model}