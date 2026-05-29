"""
Working with LLMs:
multi-providers, configuration, streaming
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith.client import _ANTHROPIC_API_KEY

load_dotenv()

def demo_init_chat_model():
    chat_model_1 = init_chat_model(
        model = "gpt-4o-mini",
        temperature = 0.7,
        streaming = True,
        max_retries = 3,
    )

    output = chat_model_1.invoke("Explain streaming parameter in one sentence. ")
    print(f"Response from chatgpt: {output.content}")

# switch model providers:
    if os.getenv(_ANTHROPIC_API_KEY):
        claude = init_chat_model(
            model = "claude-sonnet-4-5-20250929",
            temperature = 0.9,
            streaming = True,
            max_retries = 3
        )

    output = claude.invoke("Explain streaming parameter in one sentence")
    print(f"Response from claude: {output.content}")

def model_comparison():
    prompt = "Explain transformer archiecture in one sentence"

    models = {
        "gpt-4o-mini": init_chat_model(
            model = "gpt-4o-mini",
            temperature = 0.5,
            streaming = False,
        ),
        "gpt-4o": init_chat_model(
            model = "gpt-4o",
            temperature = 0.5,
            streaming = False,
        ),
        "claude": init_chat_model(
            model = "claude-sonnet-4-5-20250929",
            temperature = 0.5,
            streaming = False,
        )
    }
    print(f"Prompt: {prompt}\n")

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")

# types of messages:
# System Message - Sets the tone and boundaries for LLM
# Human Message - Actual content

def demo_messages():
    model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.9)

    # using message objects gives you more control over roles
    messages = [
        SystemMessage(content = "You are a quirky travel agent."),
        HumanMessage(content = "Find me a best spot to travel from Chennai in the month of June")

    ]

    print("Using Message objects")
    print(f"Messages: {messages[0]} | {messages[1]}")

    response = model.invoke(messages)
    print(f"Travel agent: {response.content}")

    # Multi turn conversation using message objects:
    messages.append(response) # Add models response to the conversation
    messages.append(HumanMessage(content="Na. Suggest some less explored places"))

    print("\nMulti-turn conversation")
    response = model.invoke(messages)
    print(f"Follow up response from travel agent: {response.content}")


if __name__ == "__main__":
    #demo_init_chat_model()
    #model_comparison()
    demo_messages()
