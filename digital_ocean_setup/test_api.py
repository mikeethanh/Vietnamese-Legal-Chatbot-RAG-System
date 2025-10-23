#!/usr/bin/env python3
"""
Script kiểm tra kết nối và test embedding API
"""

import requests
import json
import time
import sys
from typing import List, Dict, Any

class EmbeddingAPITester:
    """Class để test embedding API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        
    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Health check passed")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
            
    def test_embedding(self, texts: List[str]) -> bool:
        """Test embedding endpoint"""
        try:
            payload = {"texts": texts}
            response = requests.post(
                f"{self.base_url}/embed",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Embedding test passed")
                print(f"Embeddings shape: {len(result['embeddings'])} x {result['embedding_dim']}")
                return True
            else:
                print(f"❌ Embedding test failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Embedding test error: {e}")
            return False
            
    def test_similarity(self, texts1: List[str], texts2: List[str]) -> bool:
        """Test similarity endpoint"""
        try:
            payload = {
                "texts1": texts1,
                "texts2": texts2
            }
            response = requests.post(
                f"{self.base_url}/similarity",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Similarity test passed")
                print(f"Similarities shape: {result['shape']}")
                print(f"Sample similarities: {result['similarities'][0][:3]}")
                return True
            else:
                print(f"❌ Similarity test failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Similarity test error: {e}")
            return False
            
    def benchmark_performance(self, num_texts: int = 100) -> Dict[str, Any]:
        """Benchmark performance"""
        print(f"\n🚀 Running performance benchmark with {num_texts} texts...")
        
        # Generate test texts
        test_texts = [
            f"Văn bản pháp luật số {i} về quy định và thủ tục hành chính"
            for i in range(num_texts)
        ]
        
        # Test embedding performance
        start_time = time.time()
        success = self.test_embedding(test_texts)
        end_time = time.time()
        
        if success:
            duration = end_time - start_time
            throughput = num_texts / duration
            
            results = {
                "num_texts": num_texts,
                "duration_seconds": duration,
                "throughput_texts_per_second": throughput,
                "avg_time_per_text_ms": (duration / num_texts) * 1000
            }
            
            print(f"📊 Performance Results:")
            print(f"  - Duration: {duration:.2f} seconds")
            print(f"  - Throughput: {throughput:.2f} texts/second")
            print(f"  - Average time per text: {results['avg_time_per_text_ms']:.2f} ms")
            
            return results
        else:
            print("❌ Performance benchmark failed")
            return {}

def main():
    """Main test function"""
    print("🧪 Starting Vietnamese Legal Embedding API Tests")
    print("=" * 50)
    
    # Khởi tạo tester
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    tester = EmbeddingAPITester(base_url)
    
    print(f"Testing API at: {base_url}")
    print()
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    health_ok = tester.test_health()
    
    if not health_ok:
        print("❌ API is not healthy, stopping tests")
        return
        
    print()
    
    # Test 2: Embedding
    print("2. Testing embedding endpoint...")
    sample_texts = [
        "Luật Dân sự năm 2015 quy định về quyền sở hữu",
        "Bộ luật Hình sự năm 2017 về các tội phạm",
        "Luật Doanh nghiệp năm 2020 về thành lập công ty"
    ]
    
    embedding_ok = tester.test_embedding(sample_texts)
    print()
    
    # Test 3: Similarity
    print("3. Testing similarity endpoint...")
    texts1 = ["Luật Dân sự về quyền sở hữu"]
    texts2 = ["Quy định về tài sản trong Luật Dân sự", "Luật Hình sự về tội phạm"]
    
    similarity_ok = tester.test_similarity(texts1, texts2)
    print()
    
    # Test 4: Performance benchmark
    performance_ok = True
    try:
        results = tester.benchmark_performance(50)
        if results:
            # Kiểm tra performance thresholds
            if results["throughput_texts_per_second"] < 1.0:
                print("⚠️  Warning: Low throughput detected")
            if results["avg_time_per_text_ms"] > 1000:
                print("⚠️  Warning: High latency detected")
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        performance_ok = False
    
    # Tổng kết
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"  Health check: {'✅' if health_ok else '❌'}")
    print(f"  Embedding: {'✅' if embedding_ok else '❌'}")
    print(f"  Similarity: {'✅' if similarity_ok else '❌'}")
    print(f"  Performance: {'✅' if performance_ok else '❌'}")
    
    all_passed = all([health_ok, embedding_ok, similarity_ok, performance_ok])
    print(f"\n🎯 Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())