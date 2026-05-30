"""
Exercise: Create a function that:
1. Takes a question and list of modal names
2. Gets responses from all the models
3. Returns a dict of {model_name: response}

Test with: question="What is AI", models=["gpt-4o-mini", "gpt-4o"]
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def question_multi_models(question: str, model_names: list[str]):
    responses = {}
    for model_name in model_names:
        model = init_chat_model(
            model = model_name,
            temperature = 0.9,
            streaming = False,
        )
        response = model.invoke(question)
        responses[model_name] = response.content
    return responses

# Test the functions:

output = question_multi_models("What is AI?", ["gpt-4o-mini", "gpt-4o"])
for model, result in output.items():
    print(f"Response from {model}: {result}\n")

#if __name__ == "__main__":
#    question_multi_models()