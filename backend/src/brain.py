import json
import logging
import os
from openai import OpenAI
from redis import InvalidResponse

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", default=None)

def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

client = get_openai_client()

def openai_chat_complete(messages=(), model="gpt-4o-mini", raw=False):
    logger.info("Chat complete for {}".format(messages))
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    if raw:
        return response.choices[0].message
    output = response.choices[0].message
    logger.info("Chat complete output: ".format(output))
    return output.content

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def gen_doc_prompt(docs):
    """
    Document:
    Title: Uong atiso ...
    Content: ....
    """
    doc_prompt = ""
    for doc in docs:
        doc_prompt += f"Title: {doc['title']} \n Content: {doc['content']} \n"

    return "Document: \n + {}".format(doc_prompt)


def generate_conversation_text(conversations):
    conversation_text = ""
    for conversation in conversations:
        logger.info("Generate conversation: {}".format(conversation))
        role = conversation.get("role", "user")
        content = conversation.get("content", "")
        conversation_text += f"{role}: {content}\n"
    return conversation_text


def detect_user_intent(history, message):
    """
    Detect user intent and rephrase follow-up questions to standalone questions.
    Improved for Vietnamese legal context with better prompt engineering.
    """
    # Convert history to list messages
    history_messages = generate_conversation_text(history)
    logger.info(f"History messages: {history_messages}")
    
    # Check if this is likely a follow-up question
    follow_up_indicators = ["đó", "này", "kia", "thế", "vậy", "nữa", "còn", "như vậy", "như thế"]
    is_follow_up = any(indicator in message.lower() for indicator in follow_up_indicators)
    
    # If no history or not a follow-up, return original
    if not history or len(history) <= 1 and not is_follow_up:
        logger.info("No context needed, returning original query")
        return message
    
    # Update documents to prompt with better Vietnamese legal context
    user_prompt = f"""Bạn là trợ lý AI chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là viết lại câu hỏi tiếp theo thành một câu hỏi độc lập, rõ ràng và đầy đủ ngữ cảnh.

Lịch sử hội thoại:
{history_messages}

Câu hỏi hiện tại: {message}

Hướng dẫn:
1. Viết lại câu hỏi sao cho có thể hiểu được mà KHÔNG cần đọc lịch sử hội thoại
2. Thay thế các đại từ (nó, đó, này, kia, thế, vậy) bằng danh từ hoặc cụm từ cụ thể từ ngữ cảnh
3. Bổ sung thông tin cần thiết từ lịch sử để câu hỏi trở nên đầy đủ
4. Giữ nguyên ý định hỏi về pháp luật của người dùng
5. Sử dụng thuật ngữ pháp lý chính xác và phù hợp với ngữ cảnh Việt Nam
6. CHỈ trả về câu hỏi đã viết lại, KHÔNG giải thích thêm

Ví dụ:
Lịch sử: "User: Thủ tục ly hôn như thế nào?\nAssistant: Thủ tục ly hôn theo quy định..."
Câu hỏi: "Còn chi phí thì sao?"
Kết quả: "Chi phí thủ tục ly hôn theo pháp luật Việt Nam là bao nhiêu?"

Câu hỏi đã viết lại:"""

    openai_messages = [
        {"role": "system", "content": "Bạn là chuyên gia tư vấn pháp luật Việt Nam, giỏi phân tích và làm rõ câu hỏi pháp lý."},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Rephrase input messages: {openai_messages}")
    
    try:
        rephrased = openai_chat_complete(openai_messages)
        logger.info(f"Rephrased question: {rephrased}")
        return rephrased.strip()
    except Exception as e:
        logger.error(f"Error rephrasing question: {e}")
        return message



def detect_route(history, message):
    """
    Detect the appropriate tool/route for handling the user's query.
    Enhanced for Vietnamese legal chatbot with better routing options.
    
    Routes:
    - legal_rag: Questions about Vietnamese laws, regulations, legal procedures (uses RAG system)
    - web_search: Current events, recent legal changes, specific cases requiring internet search
    - general_chat: Greetings, small talk, off-topic conversations
    """
    logger.info(f"Detect route on history messages: {history}")
    
    # Format history for better context
    history_text = ""
    if history and len(history) > 1:
        for msg in history[-4:]:  # Last 2 exchanges
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                history_text += f"Người dùng: {content}\n"
            elif role == "assistant":
                history_text += f"Trợ lý: {content}\n"
    
    # Improved prompt for Vietnamese legal context
    user_prompt = f"""Bạn là hệ thống phân loại truy vấn cho chatbot tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là xác định công cụ phù hợp nhất để xử lý câu hỏi của người dùng.

Lịch sử hội thoại (nếu có):
{history_text}

Câu hỏi mới nhất của người dùng:
{message}

CÁC CÔNG CỤ KHẢ DỤNG:

1. "legal_rag" - Sử dụng khi:
   - Hỏi về luật pháp, quy định, nghị định, thông tư của Việt Nam
   - Hỏi về thủ tục pháp lý (ly hôn, thành lập doanh nghiệp, đăng ký đất đai, v.v.)
   - Hỏi về quyền và nghĩa vụ pháp lý
   - Hỏi về xử lý vi phạm, hình phạt, quy định pháp luật
   - Câu hỏi về điều khoản cụ thể trong văn bản pháp luật
   Ví dụ: "Thủ tục ly hôn như thế nào?", "Luật doanh nghiệp quy định gì về trách nhiệm của giám đốc?"

2. "web_search" - Sử dụng khi:
   - Hỏi về tin tức pháp luật gần đây, mới cập nhật
   - Hỏi về các vụ án cụ thể đang diễn ra
   - Cần thông tin thời sự hoặc thống kê hiện tại
   - Tìm kiếm văn bản pháp luật mới ban hành
   Ví dụ: "Luật mới về giao thông vừa được thông qua có gì?", "Tìm thông tin về vụ án X"

3. "general_chat" - Sử dụng khi:
   - Chào hỏi, xã giao (xin chào, cảm ơn, tạm biệt)
   - Hỏi về khả năng của chatbot
   - Câu hỏi không liên quan đến pháp luật
   - Trò chuyện chung chung
   Ví dụ: "Xin chào", "Bạn có thể giúp gì cho tôi?", "Thời tiết hôm nay thế nào?"

HƯỚNG DẪN:
- Phân tích kỹ ý định của người dùng
- Xem xét ngữ cảnh từ lịch sử hội thoại
- Ưu tiên "legal_rag" cho mọi câu hỏi về pháp luật Việt Nam
- CHỈ chọn "web_search" khi cần thông tin thời sự hoặc cập nhật gần đây
- CHỈ trả về MỘT trong ba từ khóa: "legal_rag", "web_search", hoặc "general_chat"
- KHÔNG giải thích, KHÔNG thêm bất kỳ văn bản nào khác

Phân loại:"""

    openai_messages = [
        {"role": "system", "content": "Bạn là hệ thống phân loại truy vấn chính xác cho chatbot pháp luật. Chỉ trả về một trong ba giá trị: legal_rag, web_search, general_chat"},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Routing query: {message}")
    
    try:
        route = openai_chat_complete(openai_messages).strip().lower()
        
        # Validate route
        valid_routes = ["legal_rag", "web_search", "general_chat"]
        if route not in valid_routes:
            # Try to extract valid route from response
            for valid_route in valid_routes:
                if valid_route in route:
                    route = valid_route
                    break
            else:
                # Default to legal_rag for legal chatbot
                logger.warning(f"Invalid route '{route}', defaulting to 'legal_rag'")
                route = "legal_rag"
        
        logger.info(f"Detected route: {route}")
        return route
        
    except Exception as e:
        logger.error(f"Error detecting route: {e}")
        # Default to legal_rag
        return "legal_rag"


def get_financial_tools():
    tools = []
    logger.info(f"Financial tools: {tools}")
    return tools


def get_financial_agent_answer(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()

    # Execute the chat completion request
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )

    # Attempt to extract response details
    if not resp.choices:
        logger.error("No choices available in the response.")
        return {"role": "assistant", "content": "An error occurred, please try again later."}

    choice = resp.choices[0]
    return choice


def convert_tool_calls_to_json(tool_calls):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "arguments": json.dumps(call.function.arguments),
                    "name": call.function.name
                }
            }
            for call in tool_calls
        ]
    }

def get_financial_agent_handle(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()
    choice = get_financial_agent_answer(messages, model, tools)

    resp_content = choice.message.content
    resp_tool_calls = choice.message.tool_calls
    # Prepare the assistant's message
    if resp_content:
        return resp_content

    elif resp_tool_calls:
        logger.info(f"Process the tools call: {resp_tool_calls}")
        # List to hold tool response messages
        tool_messages = []
        # Iterate through each tool call and execute the corresponding function
        for tool_call in resp_tool_calls:
            # Display the tool call details
            logger.info(f"Tool call: {tool_call.function.name}({tool_call.function.arguments})")
            # Retrieve the tool function from available tools
            tool = available_tools[tool_call.function.name]
            # Parse the arguments for the tool function
            tool_args = json.loads(tool_call.function.arguments)
            # Execute the tool function and get the result
            result = tool(**tool_args)
            tool_args['result'] = result
            # Append the tool's response to the tool_messages list
            tool_messages.append({
                "role": "tool",  # Indicate this message is from a tool
                "content": json.dumps(tool_args),  # The result of the tool function
                "tool_call_id": tool_call.id,  # The ID of the tool call
            })
        # Update the new message to get response from LLM
        # Append the tool messages to the existing messages
        # Check here: https://platform.openai.com/docs/guides/function-calling
        next_messages = messages + [convert_tool_calls_to_json(resp_tool_calls)] + tool_messages
        return get_financial_agent_handle(next_messages, model, tools)
    else:
        raise InvalidResponse(f"The response is invalid: {choice}")
