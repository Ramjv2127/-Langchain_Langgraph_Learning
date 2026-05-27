""" Langchain core concepts - LCEL & Runnables """

# Importing necessary modules:
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

def demo_basic_chain():
    """ Demonstrate a basic chain using LCEL and runnables """

    # 1. Define a prompt template:
    prompt_template = ChatPromptTemplate.from_template("You are a smart assistant. Answer in one sentence: {question}")

    # 2. Define a model and parser:
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.6)
    parser = StrOutputParser()

    # 3. Create a chain:
    basic_chain = prompt_template | model | parser

    # 4. Execute the chain with an input:
    result = basic_chain.invoke({"question": "What is runnables in langchain?"})
    print(f"Response = {result}")

    return basic_chain

if __name__ == "__main__":
    demo_basic_chain()