"""
Script để gộp các text ngắn trong file JSONL lại với nhau
Giúp tạo ra các đoạn text có độ dài phù hợp hơn cho semantic chunking
"""
import json
import argparse
from typing import List, Dict


def merge_short_texts(
    input_file: str,
    output_file: str,
    min_length: int = 100,
    max_merged_length: int = 500,
    separator: str = " "
):
    """
    Gộp các text ngắn lại với nhau
    
    Args:
        input_file: Đường dẫn file JSONL đầu vào
        output_file: Đường dẫn file JSONL đầu ra
        min_length: Độ dài tối thiểu để coi là text ngắn (ký tự)
        max_merged_length: Độ dài tối đa khi gộp các text (ký tự)
        separator: Ký tự ngăn cách giữa các text khi gộp
    """
    
    print(f"Đang đọc file: {input_file}")
    print(f"Ngưỡng độ dài tối thiểu: {min_length} ký tự")
    print(f"Độ dài tối đa khi gộp: {max_merged_length} ký tự")
    print(f"Separator: '{separator}'")
    
    merged_data = []
    buffer = []  # Buffer để lưu các text ngắn cần gộp
    
    total_lines = 0
    short_texts = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            
            if total_lines % 10000 == 0:
                print(f"Đã xử lý: {total_lines} dòng...")
            
            try:
                item = json.loads(line.strip())
                text = item.get('text', '')
                text_length = len(text)
                
                # Nếu text dài đủ, xử lý buffer trước rồi thêm text này
                if text_length >= min_length:
                    # Xử lý buffer nếu có
                    if buffer:
                        merged_text = separator.join([b['text'] for b in buffer])
                        merged_item = {
                            'id': buffer[0]['id'],
                            'text': merged_text
                        }
                        merged_data.append(merged_item)
                        buffer = []
                    
                    # Thêm text dài vào kết quả
                    merged_data.append(item)
                
                # Nếu text ngắn
                else:
                    short_texts += 1
                    buffer.append(item)
                    
                    # Kiểm tra tổng độ dài buffer
                    current_buffer_length = sum(len(b['text']) for b in buffer)
                    
                    # Nếu buffer đạt độ dài tối đa, gộp lại
                    if current_buffer_length >= max_merged_length:
                        merged_text = separator.join([b['text'] for b in buffer])
                        merged_item = {
                            'id': buffer[0]['id'],
                            'text': merged_text
                        }
                        merged_data.append(merged_item)
                        buffer = []
                        
            except json.JSONDecodeError as e:
                print(f"Lỗi decode JSON ở dòng {total_lines}: {e}")
                continue
    
    # Xử lý buffer còn lại
    if buffer:
        merged_text = separator.join([b['text'] for b in buffer])
        merged_item = {
            'id': buffer[0]['id'],
            'text': merged_text
        }
        merged_data.append(merged_item)
    
    # Ghi kết quả ra file
    print(f"\nĐang ghi kết quả vào: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Thống kê
    print("\n" + "="*50)
    print("THỐNG KÊ")
    print("="*50)
    print(f"Tổng số dòng đầu vào: {total_lines}")
    print(f"Số text ngắn (< {min_length} ký tự): {short_texts}")
    print(f"Số dòng đầu ra: {len(merged_data)}")
    print(f"Đã giảm: {total_lines - len(merged_data)} dòng ({((total_lines - len(merged_data))/total_lines*100):.2f}%)")
    
    # Phân tích độ dài text
    lengths = [len(item['text']) for item in merged_data]
    if lengths:
        print(f"\nĐộ dài text trong file đầu ra:")
        print(f"  - Min: {min(lengths)} ký tự")
        print(f"  - Max: {max(lengths)} ký tự")
        print(f"  - Trung bình: {sum(lengths)/len(lengths):.2f} ký tự")
        
        # Phân bố độ dài
        very_short = sum(1 for l in lengths if l < 50)
        short = sum(1 for l in lengths if 50 <= l < 100)
        medium = sum(1 for l in lengths if 100 <= l < 300)
        long = sum(1 for l in lengths if 300 <= l < 1000)
        very_long = sum(1 for l in lengths if l >= 1000)
        
        print(f"\nPhân bố độ dài:")
        print(f"  - Rất ngắn (< 50): {very_short} ({very_short/len(lengths)*100:.2f}%)")
        print(f"  - Ngắn (50-100): {short} ({short/len(lengths)*100:.2f}%)")
        print(f"  - Trung bình (100-300): {medium} ({medium/len(lengths)*100:.2f}%)")
        print(f"  - Dài (300-1000): {long} ({long/len(lengths)*100:.2f}%)")
        print(f"  - Rất dài (≥1000): {very_long} ({very_long/len(lengths)*100:.2f}%)")
    
    print("\nHoàn thành!")


def main():
    parser = argparse.ArgumentParser(
        description='Gộp các text ngắn trong file JSONL lại với nhau'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Đường dẫn file JSONL đầu vào'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Đường dẫn file JSONL đầu ra'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=100,
        help='Độ dài tối thiểu để coi là text ngắn (mặc định: 100)'
    )
    parser.add_argument(
        '--max-merged-length',
        type=int,
        default=500,
        help='Độ dài tối đa khi gộp các text (mặc định: 500)'
    )
    parser.add_argument(
        '--separator',
        type=str,
        default=' ',
        help='Ký tự ngăn cách giữa các text khi gộp (mặc định: khoảng trắng)'
    )
    
    args = parser.parse_args()
    
    merge_short_texts(
        input_file=args.input,
        output_file=args.output,
        min_length=args.min_length,
        max_merged_length=args.max_merged_length,
        separator=args.separator
    )


if __name__ == '__main__':
    main()
