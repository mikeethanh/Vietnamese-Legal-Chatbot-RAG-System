# Frontend - Trợ Lý Pháp Lý AI

## Tổng quan

Frontend của hệ thống Trợ lý Pháp lý AI được xây dựng bằng Streamlit với giao diện hiện đại, thân thiện và chuyên nghiệp dành riêng cho lĩnh vực pháp lý Việt Nam.

## Tính năng chính

### 🎨 Giao diện người dùng
- **Theme màu chuyên nghiệp**: Sử dụng màu xanh navy (#1e40af) làm màu chủ đạo, tạo cảm giác tin cậy và chuyên nghiệp
- **Typography tối ưu**: Font Inter và Source Sans Pro hỗ trợ tốt tiếng Việt
- **Responsive design**: Tự động điều chỉnh trên các thiết bị khác nhau
- **Dark/Light theme**: Hỗ trợ cả hai chế độ sáng và tối

### 💬 Tính năng chat
- **Streaming response**: Hiển thị phản hồi theo thời gian thực
- **Message timestamps**: Thời gian gửi/nhận tin nhắn
- **Typing indicator**: Hiển thị khi bot đang trả lời
- **Message formatting**: Hỗ trợ markdown và định dạng văn bản

### 🛠️ Chức năng quản lý
- **Clear conversation**: Xóa cuộc trò chuyện hiện tại
- **Export conversation**: Xuất cuộc trò chuyện ra file JSON
- **Conversation statistics**: Thống kê số lượng tin nhắn và câu hỏi
- **Error handling**: Xử lý lỗi một cách thân thiện

### ⚖️ Tính năng pháp lý
- **Legal categories**: Các lĩnh vực pháp lý phổ biến
- **Quick questions**: Câu hỏi mẫu cho từng lĩnh vực
- **Legal disclaimer**: Lưu ý về tính chất tham khảo của thông tin

## Files chính

### `chat_interface_new.py`
File giao diện chính với các cải tiến:
- Cấu trúc OOP sạch sẽ với class `ChatApp`
- Xử lý lỗi tốt hơn với retry logic
- UI components modular
- Responsive design
- Vietnamese legal theming

### `chat_interface.py` (legacy)
File giao diện cũ, đơn giản hơn nhưng vẫn hoạt động.

### `config.toml`
Cấu hình Streamlit:
- Theme colors và fonts
- Server settings
- UI customization
- Browser behavior

## Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng mới

```bash
streamlit run chat_interface_new.py --server.port 8051
```

### 3. Chạy với Docker

```bash
docker-compose up frontend
```

## Cấu hình

### Environment Variables
- `API_BASE_URL`: URL của backend API (mặc định: http://chatbot-api:8000)
- `BOT_ID`: ID của bot (mặc định: botFinance)
- `USER_ID`: ID của user (mặc định: 1)

### Streamlit Config
Chỉnh sửa `config.toml` để thay đổi:
- Màu sắc theme
- Font chữ
- Cấu hình server
- UI behavior

## Tính năng nâng cao

### Export Conversation
- Xuất cuộc trò chuyện ra định dạng JSON
- Bao gồm metadata và timestamps
- Có thể import lại để tiếp tục cuộc trò chuyện

### Legal Categories
Các lĩnh vực pháp lý được hỗ trợ:
- 📜 Luật Dân sự
- 🏢 Luật Doanh nghiệp  
- ⚖️ Luật Hình sự
- 🏠 Luật Đất đai
- 👥 Luật Lao động
- 📋 Luật Hành chính

### Responsive Design
- Desktop: Sidebar mở rộng, layout 2 cột
- Tablet: Sidebar thu gọn, layout tối ưu
- Mobile: Single column, touch-friendly

## Customization

### Thay đổi màu sắc
Chỉnh sửa CSS variables trong `chat_interface_new.py`:

```css
:root {
    --primary-color: #1e40af;      /* Màu chính */
    --secondary-color: #dc2626;    /* Màu phụ */
    --accent-color: #059669;       /* Màu nhấn */
    --background-color: #f8fafc;   /* Màu nền */
}
```

### Thêm tính năng mới
1. Thêm method mới trong class `ChatApp`
2. Gọi method trong `run()` function
3. Thêm CSS styling nếu cần
4. Test với các trường hợp khác nhau

## Troubleshooting

### Lỗi thường gặp

**1. Connection Error**
```
requests.exceptions.ConnectionError
```
- Kiểm tra backend API có đang chạy không
- Xác nhận URL trong cấu hình

**2. Import Error**
```
ModuleNotFoundError: No module named 'streamlit'
```
- Cài đặt lại dependencies: `pip install -r requirements.txt`

**3. Port đã được sử dụng**
```
OSError: [Errno 48] Address already in use
```
- Thay đổi port: `streamlit run app.py --server.port 8502`

### Debug mode
Chạy với debug để xem log chi tiết:
```bash
streamlit run chat_interface_new.py --logger.level debug
```

## Phát triển tiếp

### Roadmap
- [ ] Voice input/output
- [ ] Multi-language support  
- [ ] Advanced analytics
- [ ] User authentication
- [ ] Conversation history persistence
- [ ] Real-time collaboration
- [ ] Mobile app version

### Contributing
1. Fork repository
2. Tạo feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## Liên hệ
- GitHub: [Vietnamese-Legal-Chatbot-RAG-System](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System)
- Issues: Báo cáo lỗi trên GitHub Issues