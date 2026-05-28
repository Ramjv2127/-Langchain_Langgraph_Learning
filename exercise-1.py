""" 
Exercise: Create a chain that:
1. Takes a product name and target audience
2. Generates a market tagline
3. Returns just the tagline as a string

Test with: product = "AI course", audience = "developers"
"""

# Step 1: Import necessary modules.

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from pydantic.v1 import StrBytes

load_dotenv()

def exercise_first_chain():
    #Step 2: Define a prompt that takes product name and target audience
    prompt_template = ChatPromptTemplate.from_template("Generate market tagline based on '{product}' targeting '{audience}'.")
    model = ChatOpenAI(model="gpt-4o-mini", temperature = 0.7)
    parser = StrOutputParser()

    # Compose with pipe operator:
    chain = prompt_template | model | parser

    # Multiple Inputs:

    results = chain.invoke({"product": "AI course", "audience": "developers"})
    print(f"Marketing Tagline: {results}")

if __name__ == "__main__":
    exercise_first_chain()