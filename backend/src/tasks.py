import asyncio
import logging
from copy import copy
from typing import List, Dict

from splitter import split_document
from celery import shared_task

from agent import ai_agent_handle
from summarizer import summarize_text
from utils import setup_logging
from database import get_celery_app
from brain import detect_route, openai_chat_complete, detect_user_intent, get_embedding, gen_doc_prompt, \
    get_financial_agent_handle
from configs import DEFAULT_COLLECTION_NAME
from models import update_chat_conversation, get_conversation_messages
from vectorize import search_vector, add_vector
from rerank import rerank_documents
from query_rewriter import rewrite_query_to_multi_queries, rewrite_query_with_context
from search import search_engine

setup_logging()
logger = logging.getLogger(__name__)

celery_app = get_celery_app(__name__)
celery_app.autodiscover_tasks()

def follow_up_question(history, question):
    """Handle follow-up questions by rephrasing with context"""
    user_intent = detect_user_intent(history, question)
    logger.info(f"User intent (rephrased): {user_intent}")
    return user_intent


def retrieve_with_multi_query(queries: List[str], top_k: int = 5) -> List[Dict]:
    """
    Retrieve documents using multiple queries and merge results.
    This improves retrieval coverage by searching with diverse phrasings.
    
    Args:
        queries: List of query variations
        top_k: Number of documents to retrieve per query
    
    Returns:
        Merged and deduplicated list of documents
    """
    all_docs = []
    seen_contents = set()
    
    for query in queries:
        logger.info(f"Retrieving with query: {query}")
        # Get embedding for this query variation
        vector = get_embedding(query)
        
        # Search documents
        docs = search_vector(DEFAULT_COLLECTION_NAME, vector, top_k)
        
        # Deduplicate based on content
        for doc in docs:
            content_hash = hash(doc.get('content', ''))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)
    
    logger.info(f"Retrieved {len(all_docs)} unique documents from {len(queries)} queries")
    return all_docs


@shared_task()
def bot_rag_answer_message(history, question):
    """
    Enhanced RAG pipeline with:
    1. Follow-up question handling
    2. Multi-query retrieval
    3. Reranking
    4. Improved prompting for Vietnamese legal context
    """
    # Step 1: Handle follow-up questions by rephrasing with context
    standalone_question = follow_up_question(history, question)
    logger.info(f"Standalone question: {standalone_question}")
    
    # Step 2: Generate multiple query variations for better retrieval coverage
    query_variations = rewrite_query_to_multi_queries(standalone_question, num_queries=3)
    logger.info(f"Query variations: {query_variations}")
    
    # Step 3: Retrieve documents using all query variations
    retrieved_docs = retrieve_with_multi_query(query_variations, top_k=4)
    logger.info(f"Retrieved {len(retrieved_docs)} documents before reranking")
    
    # Step 4: Rerank documents based on relevance to the original question
    # Use the standalone question for reranking to ensure relevance
    ranked_docs = rerank_documents(retrieved_docs, standalone_question, top_n=5)
    logger.info(f"Top {len(ranked_docs)} documents after reranking")
    
    # Step 5: Construct enhanced prompt with legal context
    system_prompt = """Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là:
1. Trả lời câu hỏi dựa trên các tài liệu pháp luật được cung cấp
2. Trích dẫn chính xác các điều khoản, khoản, điểm từ văn bản pháp luật
3. Giải thích rõ ràng, dễ hiểu cho người không chuyên
4. Nếu thông tin không đủ trong tài liệu, hãy nói rõ điều đó
5. Luôn đưa ra câu trả lời có căn cứ pháp lý

QUAN TRỌNG: Chỉ sử dụng thông tin từ các tài liệu được cung cấp bên dưới."""

    doc_context = gen_doc_prompt(ranked_docs)
    
    openai_messages = [
        {"role": "system", "content": system_prompt}
    ] + history + [
        {
            "role": "user",
            "content": f"{doc_context}\n\nCâu hỏi: {question}\n\nHãy trả lời dựa trên các tài liệu pháp luật trên."
        }
    ]

    logger.info(f"Sending {len(openai_messages)} messages to OpenAI")

    # Step 6: Generate answer
    assistant_answer = openai_chat_complete(openai_messages)

    logger.info(f"Bot RAG reply generated successfully")
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
def bot_route_answer_message(history, question):
    """
    Route user query to appropriate handler based on intent detection.
    
    Routes:
    - legal_rag: Use RAG system for legal questions
    - web_search: Use Google search for current events
    - general_chat: Handle with simple conversation
    """
    # Detect the appropriate route
    route = detect_route(history, question)
    logger.info(f"Selected route: {route}")
    
    if route == 'legal_rag':
        # Use RAG system for legal questions
        return bot_rag_answer_message(history, question)
    
    elif route == 'web_search':
        # Use web search for current information
        logger.info("Using web search for query")
        
        # Rephrase question if it's a follow-up
        standalone_question = follow_up_question(history, question)
        
        # Search the web
        search_results = search_engine(standalone_question, top_k=5)
        
        # Generate answer based on search results
        system_prompt = """Bạn là trợ lý AI giúp tìm kiếm thông tin pháp luật trên internet. 
Hãy tổng hợp và trả lời câu hỏi dựa trên kết quả tìm kiếm được cung cấp."""
        
        openai_messages = [
            {"role": "system", "content": system_prompt}
        ] + history + [
            {
                "role": "user", 
                "content": f"Kết quả tìm kiếm:\n{search_results}\n\nCâu hỏi: {question}\n\nHãy tổng hợp thông tin và trả lời."
            }
        ]
        
        return openai_chat_complete(openai_messages)
    
    else:  # general_chat
        # Handle general conversation
        logger.info("Using general chat")
        
        system_prompt = """Bạn là trợ lý AI thân thiện của hệ thống tư vấn pháp luật Việt Nam. 
Hãy trả lời lịch sự và hướng dẫn người dùng về các câu hỏi pháp luật bạn có thể giúp đỡ."""
        
        openai_messages = [
            {"role": "system", "content": system_prompt}
        ] + history + [
            {"role": "user", "content": question}
        ]
        
        return openai_chat_complete(openai_messages)

@shared_task()
def llm_handle_message(bot_id, user_id, question):
    """
    Main message handler with intelligent routing.
    
    Flow:
    1. Save user message to conversation history
    2. Load conversation context
    3. Route to appropriate handler (RAG, web search, or general chat)
    4. Generate and save response
    """
    logger.info("Start handle message")
    
    # Update chat conversation
    conversation_id = update_chat_conversation(bot_id, user_id, question, True)
    logger.info("Conversation id: %s", conversation_id)
    
    # Convert history to list messages
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[:-1]
    
    # Use intelligent routing to handle the question
    # This will automatically choose between RAG, web search, or general chat
    response = bot_route_answer_message(history, question)
    logger.info(f"Chatbot response generated")
    
    # Summarize response for storage (optional, can be disabled if not needed)
    try:
        summarized_response = get_summarized_response(response)
    except Exception as e:
        logger.warning(f"Failed to summarize response: {e}, using original")
        summarized_response = response
    
    # Save response to history
    update_chat_conversation(bot_id, user_id, summarized_response, False)
    
    # Return full response
    return {"role": "assistant", "content": response}
