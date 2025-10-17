import asyncio
import logging
from copy import copy

from backend.src.splitter import split_document
from celery import shared_task

from summarizer import summarize_text
from utils import setup_logging
from database import get_celery_app
from brain import detect_route, openai_chat_complete, detect_user_intent, get_embedding, gen_doc_prompt, \
    get_financial_agent_handle
from configs import DEFAULT_COLLECTION_NAME
from models import update_chat_conversation, get_conversation_messages
from vectorize import search_vector, add_vector

setup_logging()
logger = logging.getLogger(__name__)

celery_app = get_celery_app(__name__)
celery_app.autodiscover_tasks()


@shared_task()
def bot_rag_answer_message(history, question):
    # Embedding text
    vector = get_embedding(question)
    logger.info(f"Get vector: {question}")

    # Search document
    top_docs = search_vector(DEFAULT_COLLECTION_NAME, vector, 2)
    logger.info(f"Top docs: {top_docs}")

    openai_messages = history + [
        {
            "role": "user",
            "content": gen_doc_prompt(top_docs)
        },
        {
            "role": "user",
            "content": question
        },
    ]

    logger.info(f"Openai messages: {openai_messages}")

    assistant_answer = openai_chat_complete(openai_messages)

    logger.info(f"Bot RAG reply: {assistant_answer}")
    return assistant_answer


def index_document_v2(id, title, content, collection_name=DEFAULT_COLLECTION_NAME):
    text = title + ' ' + content
    nodes = split_document(text)
    status_list = []
    for node in nodes:
        vector = get_embedding(node.text)
        add_vector_status = add_vector(
            collection_name=collection_name,
            vectors={
                id: {
                    "vector": vector,
                    "payload": {
                        "title": title,
                        "content": node.text
                    }
                }
            }
        )
        status_list.append(add_vector_status)
    logger.info(f"Add vector status: {status_list}")
    return status_list


def get_summarized_response(response):
    output = summarize_text(response)
    logger.info("Summarized response: %s", output)
    return output


@shared_task()
def llm_handle_message(bot_id, user_id, question):
    logger.info("Start handle message")
    # Update chat conversation
    conversation_id = update_chat_conversation(bot_id, user_id, question, True)
    logger.info("Conversation id: %s", conversation_id)
    # Convert history to list messages
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[:-1]
    # Bot generation
    response = bot_rag_answer_message(history, question)
    logger.info(f"Chatbot response: {response}")
    # Summarize response
    summarized_response = get_summarized_response(response)
    # Save response to history
    update_chat_conversation(bot_id, user_id, summarized_response, False)
    # Return response
    return {"role": "assistant", "content": response}
