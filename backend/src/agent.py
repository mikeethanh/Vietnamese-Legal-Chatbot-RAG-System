import json
import logging
from typing import Dict

from celery import shared_task
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.openai import OpenAI

from legal_tools import (
    calculate_contract_penalty,
    calculate_inheritance_share,
    check_business_name_rules,
    check_legal_entity_age,
    get_statute_of_limitations,
)
from search import search_engine
from tavily_tool import tavily_qna, tavily_search_legal

logger = logging.getLogger(__name__)


# ===== LEGAL CALCULATION TOOLS =====


def contract_penalty_calculator(
    contract_value: float, penalty_rate: float, days_late: int
) -> str:
    """
    Tính tiền phạt vi phạm hợp đồng theo Bộ luật Dân sự Việt Nam.

    Args:
        contract_value: Giá trị hợp đồng (VNĐ), ví dụ: 100000000 (100 triệu)
        penalty_rate: Tỷ lệ phạt theo hợp đồng (% mỗi ngày), ví dụ: 0.1 (0.1%/ngày)
        days_late: Số ngày chậm trễ, ví dụ: 30

    Returns:
        Kết quả tính toán tiền phạt chi tiết
    """
    result = calculate_contract_penalty(contract_value, penalty_rate, days_late)
    return json.dumps(result, ensure_ascii=False, indent=2)


def legal_age_checker(birth_year: int, action_type: str = "sign_contract") -> str:
    """
    Kiểm tra tuổi pháp lý để thực hiện hành vi dân sự.

    Args:
        birth_year: Năm sinh, ví dụ: 2005
        action_type: Loại hành vi, có thể là: "sign_contract" (ký hợp đồng), "marriage" (kết hôn), "work" (làm việc), "criminal_responsibility" (chịu trách nhiệm hình sự)

    Returns:
        Thông tin về khả năng pháp lý và căn cứ pháp luật
    """
    result = check_legal_entity_age(birth_year, action_type)
    return json.dumps(result, ensure_ascii=False, indent=2)


def inheritance_calculator(total_value: float, heirs_json: str) -> str:
    """
    Tính phần thừa kế theo pháp luật Việt Nam (hàng thừa kế thứ nhất).

    Args:
        total_value: Tổng giá trị tài sản thừa kế (VNĐ), ví dụ: 500000000 (500 triệu)
        heirs_json: Danh sách người thừa kế dạng JSON string, ví dụ: '[{"name":"Nguyễn Văn A","relation":"child","is_minor":false},{"name":"Trần Thị B","relation":"spouse","is_minor":false}]'

    Returns:
        Phân chia tài sản thừa kế cho từng người
    """
    try:
        heirs = json.loads(heirs_json)
        result = calculate_inheritance_share(total_value, heirs)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return json.dumps(
            {"error": "heirs_json không đúng định dạng JSON"}, ensure_ascii=False
        )


def business_name_validator(business_name: str) -> str:
    """
    Kiểm tra tên doanh nghiệp có hợp lệ theo Luật Doanh nghiệp Việt Nam.

    Args:
        business_name: Tên doanh nghiệp cần kiểm tra, ví dụ: "Công ty TNHH ABC"

    Returns:
        Kết quả kiểm tra tính hợp lệ và các lưu ý
    """
    result = check_business_name_rules(business_name)
    return json.dumps(result, ensure_ascii=False, indent=2)


def statute_lookup(case_type: str) -> str:
    """
    Tra cứu thời hiệu khởi kiện theo pháp luật Việt Nam.

    Args:
        case_type: Loại vụ việc, có thể là: "civil" (dân sự), "labor" (lao động), "administrative" (hành chính), "criminal" (hình sự)

    Returns:
        Thông tin về thời hiệu và căn cứ pháp lý
    """
    result = get_statute_of_limitations(case_type)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ===== WEB SEARCH TOOLS =====


def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Tìm kiếm thông tin pháp luật trên internet sử dụng Google Search.
    Dùng khi cần tìm tin tức, văn bản pháp luật mới, hoặc thông tin cập nhật.

    Args:
        query: Từ khóa tìm kiếm, ví dụ: "Luật Đất đai 2024 sửa đổi"
        max_results: Số kết quả tối đa (mặc định 5)

    Returns:
        Kết quả tìm kiếm với tiêu đề, link và nội dung tóm tắt
    """
    try:
        return search_engine(query, top_k=max_results)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Lỗi tìm kiếm: {str(e)}"


def tavily_search_tool(query: str, max_results: int = 5) -> str:
    """
    Tìm kiếm thông tin pháp luật sử dụng Tavily AI (tìm kiếm thông minh với AI).
    Tavily cung cấp kết quả tốt hơn và tóm tắt tự động cho câu hỏi pháp lý.

    Args:
        query: Câu hỏi hoặc từ khóa tìm kiếm, ví dụ: "Quy định mới về BHXH 2024"
        max_results: Số kết quả tối đa (mặc định 5)

    Returns:
        Kết quả tìm kiếm với tóm tắt AI và các nguồn liên quan
    """
    try:
        return tavily_search_legal(query, max_results=max_results)
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Tavily không khả dụng: {str(e)}"


def quick_answer_tool(question: str) -> str:
    """
    Trả lời nhanh câu hỏi bằng tìm kiếm web (Tavily Q&A).
    Dùng cho câu hỏi sự kiện, thống kê, hoặc thông tin cập nhật cần web search.

    Args:
        question: Câu hỏi cần trả lời, ví dụ: "Mức lương tối thiểu vùng 1 năm 2024"

    Returns:
        Câu trả lời trực tiếp từ web search
    """
    try:
        return tavily_qna(question)
    except Exception as e:
        logger.error(f"Quick answer error: {e}")
        return f"Không thể trả lời: {str(e)}"


# ===== CREATE TOOLS =====

# Legal calculation tools
contract_penalty_tool = FunctionTool.from_defaults(fn=contract_penalty_calculator)
legal_age_tool = FunctionTool.from_defaults(fn=legal_age_checker)
inheritance_tool = FunctionTool.from_defaults(fn=inheritance_calculator)
business_name_tool = FunctionTool.from_defaults(fn=business_name_validator)
statute_tool = FunctionTool.from_defaults(fn=statute_lookup)

# Web search tools
google_search_tool = FunctionTool.from_defaults(fn=web_search_tool)
tavily_tool = FunctionTool.from_defaults(fn=tavily_search_tool)
quick_answer_tool_func = FunctionTool.from_defaults(fn=quick_answer_tool)

# All available tools for agent
all_tools = [
    # Legal tools
    contract_penalty_tool,
    legal_age_tool,
    inheritance_tool,
    business_name_tool,
    statute_tool,
    # Search tools
    google_search_tool,
    tavily_tool,
    quick_answer_tool_func,
]

# Initialize LLM and Agent
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

# Create agent with Vietnamese legal context
agent_system_prompt = """Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam với khả năng sử dụng các công cụ.

NHIỆM VỤ:
1. Trả lời câu hỏi pháp luật chính xác dựa trên các công cụ có sẵn
2. Sử dụng công cụ phù hợp cho từng loại câu hỏi
3. Giải thích rõ ràng kết quả từ công cụ cho người dùng

CÁC CÔNG CỤ KHẢ DỤNG:
- Tính toán pháp lý: phạt hợp đồng, thừa kế, tuổi pháp lý
- Kiểm tra: tên doanh nghiệp, thời hiệu khởi kiện
- Tìm kiếm web: Google Search và Tavily AI cho thông tin mới

HƯỚNG DẪN SỬ DỤNG CÔNG CỤ:
- Khi cần tính toán số liệu → dùng công cụ tính toán
- Khi cần kiểm tra quy định cụ thể → dùng công cụ tra cứu
- Khi cần thông tin mới/cập nhật → dùng công cụ tìm kiếm web
- Sau khi có kết quả từ công cụ → giải thích bằng tiếng Việt dễ hiểu

LƯU Ý:
- Luôn trích dẫn căn cứ pháp lý từ kết quả công cụ
- Nếu công cụ trả về lỗi, giải thích và đề xuất giải pháp khác
- Trả lời chính xác, chuyên nghiệp, dễ hiểu"""

ai_agent = ReActAgent.from_tools(
    all_tools, llm=llm, verbose=True, max_iterations=10, context=agent_system_prompt
)

logger.info(f"Agent initialized with {len(all_tools)} tools")


@shared_task()
def ai_agent_handle(question: str) -> str:
    """
    Handle user question using ReAct agent with tools.

    Args:
        question: User's question

    Returns:
        Agent's response
    """
    try:
        logger.info(f"[AGENT] Processing question: {question}")
        response = ai_agent.chat(question)
        logger.info(f"[AGENT] Response generated successfully")
        return response.response
    except Exception as e:
        logger.error(f"[AGENT] Error: {e}")
        return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"


def get_agent_tools_summary() -> Dict:
    """Get summary of available agent tools"""
    return {
        "total_tools": len(all_tools),
        "legal_tools": [
            "contract_penalty_calculator - Tính phạt hợp đồng",
            "legal_age_checker - Kiểm tra tuổi pháp lý",
            "inheritance_calculator - Tính thừa kế",
            "business_name_validator - Kiểm tra tên DN",
            "statute_lookup - Tra cứu thời hiệu",
        ],
        "search_tools": [
            "google_search_tool - Tìm kiếm Google",
            "tavily_search_tool - Tìm kiếm Tavily AI",
            "quick_answer_tool - Trả lời nhanh",
        ],
    }
