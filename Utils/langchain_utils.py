from langchain_core.output_parsers import StrOutputParser
from Utils.prompts import prompt_template


parser = StrOutputParser()

def get_chain(llm):
    return prompt_template | llm | parser
