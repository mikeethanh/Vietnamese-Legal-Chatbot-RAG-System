import logging
from typing import List

from brain import openai_chat_complete

logger = logging.getLogger(__name__)


def rewrite_query_to_multi_queries(
    original_query: str, num_queries: int = 3
) -> List[str]:
    """
    Rewrite a single query into multiple diverse queries for better retrieval coverage.

    This technique helps capture different aspects and perspectives of the user's question,
    improving the chances of retrieving relevant legal documents.

    Args:
        original_query: The original user question
        num_queries: Number of diverse queries to generate (default: 3)

    Returns:
        List of rewritten queries including the original
    """

    prompt = f"""Bạn là trợ lý AI chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là tạo ra {num_queries} câu hỏi khác nhau nhưng có cùng ý nghĩa với câu hỏi gốc để tìm kiếm thông tin pháp luật hiệu quả hơn.

Câu hỏi gốc: {original_query}

Yêu cầu:
1. Mỗi câu hỏi cần diễn đạt theo cách khác nhau
2. Giữ nguyên ý nghĩa và mục đích của câu hỏi gốc
3. Sử dụng từ ngữ pháp lý phù hợp với ngữ cảnh luật Việt Nam
4. Mỗi câu hỏi trên một dòng riêng biệt
5. KHÔNG đánh số thứ tự, KHÔNG giải thích, CHỈ trả về {num_queries} câu hỏi

Ví dụ:
Câu hỏi gốc: "Thủ tục ly hôn như thế nào?"
Kết quả:
Quy trình giải quyết ly hôn theo pháp luật Việt Nam
Các bước tiến hành thủ tục chấm dứt hôn nhân
Hồ sơ và trình tự ly hôn được quy định ra sao

Bây giờ hãy tạo {num_queries} câu hỏi cho câu hỏi gốc trên:"""

    messages = [
        {
            "role": "system",
            "content": "Bạn là chuyên gia tư vấn pháp luật Việt Nam, giỏi diễn đạt câu hỏi pháp lý.",
        },
        {"role": "user", "content": prompt},
    ]

    logger.info(f"Rewriting query: {original_query}")

    try:
        response = openai_chat_complete(messages)

        # Parse the response to extract queries
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering if present (1., 2., etc.)
            if line and not line.startswith("#"):
                # Remove leading numbers and dots
                cleaned_line = line.lstrip("0123456789.-) ").strip()
                if cleaned_line:
                    queries.append(cleaned_line)

        # Ensure we have the requested number of queries
        if len(queries) < num_queries:
            logger.warning(
                f"Only generated {len(queries)} queries, expected {num_queries}"
            )
            # Add original query if we don't have enough
            while len(queries) < num_queries:
                queries.append(original_query)

        # Limit to requested number
        queries = queries[:num_queries]

        logger.info(f"Generated {len(queries)} queries: {queries}")
        return queries

    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        # Fallback to original query repeated
        return [original_query] * num_queries


def rewrite_query_with_context(query: str, conversation_history: List[dict]) -> str:
    """
    Rewrite a query by incorporating conversation context.
    This helps resolve references and ambiguities in follow-up questions.

    Args:
        query: The current user query
        conversation_history: List of previous conversation messages

    Returns:
        Rewritten standalone query
    """
    if not conversation_history or len(conversation_history) <= 1:
        # No context, return original
        return query

    # Format conversation history
    history_text = ""
    for msg in conversation_history[-6:]:  # Last 3 exchanges (6 messages)
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            history_text += f"Người dùng: {content}\n"
        elif role == "assistant":
            history_text += f"Trợ lý: {content}\n"

    prompt = f"""Dựa vào lịch sử hội thoại dưới đây, hãy viết lại câu hỏi hiện tại thành một câu hỏi độc lập, rõ ràng và đầy đủ ý nghĩa bằng tiếng Việt.

Lịch sử hội thoại:
{history_text}

Câu hỏi hiện tại: {query}

Yêu cầu:
- Viết lại câu hỏi sao cho có thể hiểu được mà không cần biết ngữ cảnh trước đó
- Thay thế các đại từ (nó, đó, cái kia, v.v.) bằng danh từ cụ thể
- Giữ nguyên ý định và mục đích của câu hỏi
- CHỈ trả về câu hỏi đã viết lại, KHÔNG giải thích

Câu hỏi đã viết lại:"""

    messages = [
        {
            "role": "system",
            "content": "Bạn là trợ lý AI chuyên viết lại câu hỏi để làm rõ ngữ nghĩa.",
        },
        {"role": "user", "content": prompt},
    ]

    logger.info(f"Rewriting query with context: {query}")

    try:
        rewritten = openai_chat_complete(messages)
        logger.info(f"Rewritten query: {rewritten}")
        return rewritten.strip()
    except Exception as e:
        logger.error(f"Error rewriting query with context: {e}")
        return query
