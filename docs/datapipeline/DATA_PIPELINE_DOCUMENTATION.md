# Tài liệu Data Pipeline - Vietnamese Legal Chatbot RAG System

## 📋 Tổng quan

Data Pipeline của hệ thống Vietnamese Legal Chatbot RAG có nhiệm vụ xử lý và chuẩn bị dữ liệu từ các nguồn khác nhau để phục vụ cho việc training và triển khai chatbot tư vấn pháp luật Việt Nam.

## 🎯 Vấn đề cần giải quyết

### Bài toán chia dữ liệu
Dữ liệu cần được chia thành **3 phần chính**:

1. **📚 Dữ liệu Finetune**: Để training mô hình hiểu và trả lời câu hỏi pháp luật
2. **🔍 Dữ liệu Embedding**: Để tạo vector representations cho tìm kiếm ngữ nghĩa  
3. **💾 Dữ liệu RAG**: Để xây dựng knowledge base cho hệ thống Retrieval-Augmented Generation

### Nguồn dữ liệu
Pipeline xử lý dữ liệu từ **nhiều nguồn khác nhau**:

#### **Dữ liệu Finetune** (từ Hugging Face):
1. **`phuocsang/hoidap-tvpl-20k`** - 20k cặp hỏi đáp pháp luật tiếng Việt (process_finetune_data.ipynb)
2. **`huyhuy123/ViLQA`** - Vietnamese Legal Q&A dataset (process_finetune_data_2.ipynb)  
3. **`chillies/vn-legal-conversation`** - Vietnamese legal conversation data (process_finetune_data_3.ipynb)

#### **Dữ liệu RAG/Embedding** (từ Kaggle):
- **`anti-ai/ViNLI-Zalo-supervised`** - Vietnamese legal corpus từ file `law_vi.jsonl.gz` (download_embed_data.ipynb)

**Thách thức**: Dữ liệu từ nhiều nguồn khác nhau có format và cấu trúc khác nhau cần được tổng hợp và chuẩn hóa về một định dạng thống nhất.

## 🔧 Chi tiết các module xử lý

### 1. Module Xử lý Dữ liệu Finetune

#### 📁 File: `process_finetune_data.ipynb`

Đây là module quan trọng nhất trong pipeline, có nhiệm vụ chuyển đổi dữ liệu thô thành định dạng Q&A phù hợp cho việc training chatbot.

#### 🔍 **Phân tích dữ liệu (Data Analysis)**

**Hàm `analyze_text_quality(dataset_split, split_name)`**

```python
def analyze_text_quality(dataset_split, split_name):
    """
    Phân tích chất lượng text trong dataset
    
    Args:
        dataset_split: Phần dữ liệu cần phân tích (train/test)
        split_name: Tên của phần dữ liệu để hiển thị
        
    Returns:
        dict: Thống kê chi tiết về chất lượng dữ liệu
    """
```

**Ket qua**
📈 Phân tích Dataset ViLQA (43588 samples):
🔸 Độ dài câu hỏi:
   - Trung bình: 75.7 ký tự
   - Min: 0, Max: 263
   - Median: 71.0
🔸 Độ dài câu trả lời:
   - Trung bình: 888.6 ký tự
   - Min: 0, Max: 20674
   - Median: 673.0
🔸 Dữ liệu rỗng:
   - Câu hỏi rỗng: 48
   - Câu trả lời rỗng: 115
🔸 Câu hỏi có dấu '?': 42502/43588 (97.5%)

**Tại sao cần phân tích:**
- Hiểu được đặc điểm của dữ liệu trước khi xử lý
- Thiết lập các ngưỡng lọc dữ liệu hợp lý
- Phát hiện các vấn đề tiềm ẩn trong dataset

#### 🧹 **Làm sạch dữ liệu (Data Cleaning)**

**Hàm `clean_text(text)`**

```python
def clean_text(text):
    """
    Làm sạch và chuẩn hóa text
    
    Args:
        text (str): Text cần làm sạch
        
    Returns:
        str: Text đã được làm sạch
    """
```

**Các bước xử lý:**
1. **Loại bỏ khoảng trắng thừa**: Sử dụng `" ".join(text.split())` để normalize spaces
2. **Chuẩn hóa ký tự xuống dòng**: Thay thế `\n`, `\r`, `\t` bằng space
3. **Trim space**: Loại bỏ space đầu và cuối chuỗi

**Tại sao cần làm sạch:**
- Đảm bảo tính nhất quán trong format
- Loại bỏ noise có thể ảnh hưởng đến chất lượng training
- Chuẩn hóa để dễ dàng xử lý sau này

#### 🎯 **Lọc và xử lý dữ liệu (Data Filtering)**

**Hàm `process_dataset(dataset_split, max_answer_length=5000)`**

```python
def process_dataset(dataset_split, max_answer_length=5000):
    """
    Xử lý dataset và lọc dữ liệu chất lượng
    
    Args:
        dataset_split: Dataset cần xử lý
        max_answer_length (int): Độ dài tối đa của câu trả lời
        
    Returns:
        list: Danh sách các mẫu dữ liệu đã được lọc và xử lý
    """
```
**ket qua**
✅ Dataset processed: 43588 → 43420 (giữ lại 99.6%)

📊 Thống kê sau xử lý:
- Tổng số mẫu chất lượng: 43420

🔸 Độ dài câu hỏi sau xử lý:
   - Trung bình: 75.9 ký tự
   - Min: 10, Max: 263
🔸 Độ dài câu trả lời sau xử lý:
   - Trung bình: 882.7 ký tự
   - Min: 51, Max: 7981

**Tiêu chí lọc:**
- **Độ dài câu hỏi tối thiểu**: >= 10 ký tự (đảm bảo câu hỏi có ý nghĩa)
- **Độ dài câu trả lời tối thiểu**: >= 50 ký tự (đảm bảo câu trả lời đầy đủ)
- **Độ dài câu trả lời tối đa**: <= 5000 ký tự (tránh context quá dài)
- **Format câu hỏi**: Phải kết thúc bằng dấu '?' (đảm bảo là câu hỏi thực sự)

**Lý do các tiêu chí:**
- Đảm bảo chất lượng dữ liệu training
- Tránh overfitting với các mẫu không chuẩn
- Tối ưu hóa hiệu suất training và inference

### 2. Module Lưu trữ dữ liệu (Data Storage)

#### 📦 **Định dạng lưu trữ đa dạng**

Pipeline hỗ trợ **2 định dạng** chính để phù hợp với các mục đích training khác nhau:

#### **Format 1: QA Format (Question-Answer)**

**Hàm `save_jsonl(data, filepath)`**

```python
def save_jsonl(data, filepath):
    """
    Lưu dữ liệu dưới định dạng JSONL cơ bản
    
    Structure:
    {
        "question": "Câu hỏi pháp luật",
        "answer": "Câu trả lời chi tiết"
    }
    """
```

**Sử dụng cho:**
- Traditional Q&A training
- Simple fine-tuning approaches
- Evaluation và testing

#### **Format 2: Instruction Format**

**Hàm `save_instruction_format(data, filepath)`**

```python
def save_instruction_format(data, filepath):
    """
    Lưu dữ liệu dưới định dạng instruction tuning
    
    Structure:
    {
        "instruction": "Trả lời câu hỏi pháp luật sau:",
        "input": "Câu hỏi của user",
        "output": "Câu trả lời mong muốn"
    }
    """
```

**Tại sao cần Instruction Format:**
- **Tính nhất quán**: Mô hình học được cách tuân theo instructions
- **Khả năng generalization**: Mô hình có thể áp dụng cho các loại instructions khác
- **Chất lượng output**: Cải thiện độ chính xác và relevance của câu trả lời
- **Alignment**: Đảm bảo mô hình tuân theo human preference

### 3. Module Metadata và Validation

#### 📊 **Tạo Metadata**

Pipeline tự động tạo metadata chi tiết bao gồm:

```json
{
    "dataset_info": {
        "source": "phuocsang/hoidap-tvpl-20k",
        "description": "Vietnamese Legal Q&A Dataset processed for fine-tuning",
        "total_samples": "Tổng số mẫu",
        "train_samples": "Số mẫu train",
        "test_samples": "Số mẫu test"
    },
    "processing_info": {
        "filters_applied": ["Danh sách các bộ lọc đã áp dụng"],
        "retention_rate": "Tỷ lệ dữ liệu được giữ lại"
    },
    "file_formats": {
        "qa_format": "Mô tả format",
        "instruction_format": "Mô tả format", 
        "conversation_format": "Mô tả format"
    }
}
```

#### ✅ **Validation dữ liệu**

**Hàm `validate_jsonl_file(filepath, expected_count)`**

```python
def validate_jsonl_file(filepath, expected_count):
    """
    Kiểm tra tính toàn vẹn của file JSONL
    
    Validates:
    - JSON format correctness
    - Expected number of records
    - File readability
    """
```

**Kiểm tra:**
- Tính hợp lệ của JSON format
- Số lượng records matches expected
- Khả năng đọc file
- Encoding UTF-8 đúng chuẩn

## 🔄 Workflow tổng thể

```
1. Load Dataset từ Hugging Face
    ↓
2. Phân tích chất lượng dữ liệu (Analysis)
    ↓
3. Làm sạch text (Cleaning) 
    ↓
4. Lọc theo tiêu chí chất lượng (Filtering)
    ↓
5. Chuyển đổi sang multiple formats (Transformation)
    ↓
6. Lưu trữ với metadata (Storage)
    ↓
7. Validation và quality check (Validation)
```

*Tài liệu này mô tả chi tiết architecture và implementation của Data Pipeline trong Vietnamese Legal Chatbot RAG System. Để biết thêm chi tiết về implementation cụ thể, vui lòng tham khảo source code trong thư mục `data_pipeline/`.*