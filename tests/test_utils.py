"""
Test utilities module
"""
import json
import tempfile
from typing import Dict, Any, List
from unittest.mock import Mock

def create_mock_response(data: Dict[str, Any], status_code: int = 200) -> Mock:
    """Create a mock HTTP response"""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = data
    mock_response.text = json.dumps(data)
    return mock_response

def create_test_qa_data() -> List[Dict[str, Any]]:
    """Create test Q&A data for testing"""
    return [
        {
            "question": "Luật giao thông quy định gì về tốc độ tối đa?",
            "answer": "Theo Luật Giao thông đường bộ, tốc độ tối đa trong khu vực đông dân cư là 50km/h.",
            "metadata": {
                "law": "Luật Giao thông đường bộ",
                "article": "Điều 16",
                "category": "giao_thong"
            }
        },
        {
            "question": "Quy định về hợp đồng lao động là gì?",
            "answer": "Hợp đồng lao động phải được ký kết bằng văn bản và có các điều khoản cơ bản theo quy định.",
            "metadata": {
                "law": "Bộ luật Lao động",
                "article": "Điều 15",
                "category": "lao_dong"
            }
        }
    ]

def create_temp_json_file(data: List[Dict[str, Any]]) -> str:
    """Create a temporary JSON file with test data"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, temp_file, ensure_ascii=False, indent=2)
    temp_file.close()
    return temp_file.name

def assert_valid_response_format(response: Dict[str, Any]):
    """Assert that response has valid format"""
    assert "answer" in response
    assert "sources" in response
    assert isinstance(response["answer"], str)
    assert isinstance(response["sources"], list)
    assert len(response["answer"]) > 0