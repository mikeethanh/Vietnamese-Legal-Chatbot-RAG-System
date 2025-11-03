"""
Test utilities module - standalone functions without external dependencies
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
    import tempfile
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

# Test the utils themselves
def test_create_mock_response():
    """Test the mock response creation"""
    data = {"test": "value"}
    mock_resp = create_mock_response(data, 200)
    assert mock_resp.status_code == 200
    assert mock_resp.json() == data

def test_create_test_qa_data():
    """Test the test data creation"""
    data = create_test_qa_data()
    assert len(data) == 2
    assert all("question" in item for item in data)
    assert all("answer" in item for item in data)

def test_assert_valid_response_format():
    """Test the response format validator"""
    valid_response = {
        "answer": "Test answer",
        "sources": ["Source 1", "Source 2"]
    }
    # Should not raise an exception
    assert_valid_response_format(valid_response)