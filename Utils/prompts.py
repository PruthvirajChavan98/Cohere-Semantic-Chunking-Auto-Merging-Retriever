from langchain_core.prompts import  SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

system_prompt = SystemMessagePromptTemplate.from_template(
        template="""
       - you are assitant that helps with context provided from a pdf
       - you MUST preserve context and highlight answer in pointwise manner
        """
    )

human_template = HumanMessagePromptTemplate.from_template(
    template="{text}\n\nfrom this context highlight information on:\n{topic}", input_variables=["text", "topic"]
    )

prompt_template = ChatPromptTemplate.from_messages(
    [
        human_template,
        system_prompt
    ]
)