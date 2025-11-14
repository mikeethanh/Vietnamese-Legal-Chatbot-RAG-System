"""
Script để test Vietnamese Legal LLM API
"""

import requests
import json
import time
from typing import Dict, List, Any

class LegalLLMTester:
    def __init__(self, base_url: str = "http://localhost:6000"):
        self.base_url = base_url
        
    def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        print("🏥 Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Model loaded: {health_data.get('model_loaded')}")
            print(f"   GPU available: {health_data.get('gpu_available')}")
            
            if health_data.get('memory_usage'):
                memory = health_data['memory_usage']
                print(f"   GPU memory: {memory.get('gpu_memory_allocated', 0):.1f}GB allocated")
            
            return health_data
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {}
    
    def test_chat_completion(self, question: str) -> Dict[str, Any]:
        """Test chat completion endpoint"""
        print(f"💬 Testing chat completion...")
        print(f"   Question: {question}")
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response
            assistant_message = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            
            print(f"✅ Chat completion successful")
            print(f"   Response time: {end_time - start_time:.2f}s")
            print(f"   Tokens used: {usage.get('total_tokens', 'N/A')}")
            print(f"   Response: {assistant_message[:200]}...")
            
            return result
            
        except Exception as e:
            print(f"❌ Chat completion failed: {e}")
            return {}
    
    def test_streaming(self, question: str):
        """Test streaming endpoint"""
        print(f"🌊 Testing streaming...")
        print(f"   Question: {question}")
        
        payload = {
            "messages": [
                {
                    "role": "user", 
                    "content": question
                }
            ],
            "temperature": 0.7,
            "max_tokens": 300,
            "stream": True
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions/stream",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            response.raise_for_status()
            
            print(f"✅ Streaming started")
            print(f"   Response: ", end="", flush=True)
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_text = line_text[6:]  # Remove 'data: ' prefix
                        
                        if data_text == '[DONE]':
                            break
                            
                        try:
                            chunk_data = json.loads(data_text)
                            delta = chunk_data.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            
                            if content:
                                print(content, end="", flush=True)
                                full_response += content
                                
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            print(f"\n   Stream completed in {end_time - start_time:.2f}s")
            print(f"   Total length: {len(full_response)} characters")
            
        except Exception as e:
            print(f"❌ Streaming failed: {e}")
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("=" * 80)
        print("🧪 VIETNAMESE LEGAL LLM API TEST SUITE")
        print("=" * 80)
        
        # Test cases
        test_questions = [
            "Quy định về thời hiệu khởi kiện trong vụ việc dân sự là gì?",
            "Người lao động có quyền nghỉ việc riêng bao nhiêu ngày trong năm?",
            "Điều kiện để thành lập doanh nghiệp tư nhân là gì?",
            "Xử phạt vi phạm giao thông đối với việc vượt đèn đỏ như thế nào?",
            "Quy trình ly hôn thuận tình được thực hiện ra sao?"
        ]
        
        # 1. Health check
        health = self.test_health()
        if not health.get('model_loaded'):
            print("❌ Model not loaded, stopping tests")
            return
            
        print("\n" + "-" * 50)
        
        # 2. Test regular chat completions
        for i, question in enumerate(test_questions[:3], 1):
            print(f"\n📝 Test Case {i}/3:")
            self.test_chat_completion(question)
            time.sleep(1)  # Rate limiting
        
        print("\n" + "-" * 50)
        
        # 3. Test streaming
        print(f"\n🌊 Streaming Test:")
        self.test_streaming(test_questions[3])
        
        print("\n" + "-" * 50)
        
        # 4. Test models endpoint
        print(f"\n📋 Testing models endpoint...")
        try:
            response = requests.get(f"{self.base_url}/models")
            response.raise_for_status()
            models = response.json()
            print(f"✅ Models endpoint working")
            print(f"   Available models: {len(models.get('data', []))}")
        except Exception as e:
            print(f"❌ Models endpoint failed: {e}")
        
        print("\n" + "=" * 80)
        print("✅ TEST SUITE COMPLETED")
        print("=" * 80)

def main():
    # Test different endpoints
    base_urls = [
        "http://localhost:6000",      # Local development
        "http://0.0.0.0:6000",       # Docker local
        # "http://your-droplet-ip:6000"  # Digital Ocean
    ]
    
    for base_url in base_urls:
        print(f"\n🔍 Testing: {base_url}")
        tester = LegalLLMTester(base_url)
        
        try:
            # Quick health check first
            health = tester.test_health()
            if health.get('status') == 'healthy':
                # Run one quick test
                tester.test_chat_completion("Xin chào, bạn có thể giúp tôi tư vấn pháp luật không?")
                break
            else:
                print(f"⏭️  Skipping {base_url} - service not healthy")
        except Exception as e:
            print(f"⏭️  Skipping {base_url} - connection failed: {e}")
    
    # Interactive mode
    print(f"\n🤖 Interactive mode (Ctrl+C to exit)")
    tester = LegalLLMTester()
    
    try:
        while True:
            question = input("\n❓ Enter your legal question: ")
            if question.lower() in ['exit', 'quit', 'bye']:
                break
                
            tester.test_chat_completion(question)
    except KeyboardInterrupt:
        print(f"\n👋 Goodbye!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # Run comprehensive test
        tester = LegalLLMTester()
        tester.run_comprehensive_test()
    else:
        # Run basic test
        main()