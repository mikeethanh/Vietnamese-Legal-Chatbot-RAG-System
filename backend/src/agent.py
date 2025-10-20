import logging

from celery import shared_task
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from search import search_engine

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
search_tool = FunctionTool.from_defaults(fn=search_engine)

llm = OpenAI(model="gpt-4o-mini")
ai_agent = ReActAgent.from_tools([multiply_tool, add_tool, search_tool], llm=llm, verbose=True)

@shared_task()
def ai_agent_handle(question):
    response = ai_agent.chat(question)
    logging.info(f"Agent response: {response}")
    return response.response
