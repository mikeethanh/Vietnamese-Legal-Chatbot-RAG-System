import json
import logging
import time
import datetime
from typing import List, Dict, Any
from io import StringIO

import requests
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trợ Lý Pháp Lý AI - Vietnamese Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern Vietnamese legal theme
st.markdown("""
<style>
    /* Import Vietnamese-friendly fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    /* Main theme colors - Vietnamese legal colors */
    :root {
        --primary-color: #1e40af;
        --secondary-color: #dc2626;
        --accent-color: #059669;
        --background-color: #f8fafc;
        --surface-color: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, #3b82f6 100%);
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(30, 64, 175, 0.15);
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin: 0;
        font-size: 2rem;
        text-align: center;
    }
    
    .main-header .subtitle {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--surface-color);
        border-right: 2px solid var(--border-color);
    }
    
    /* Chat container */
    .chat-container {
        background-color: var(--surface-color);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border-color);
    }
    
    /* Message styling */
    .stChatMessage {
        background-color: var(--surface-color);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
        font-family: 'Source Sans Pro', sans-serif;
        line-height: 1.6;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, #3b82f6 100%);
        color: white;
        margin-left: 20%;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #f1f5f9;
        border-left: 4px solid var(--accent-color);
        margin-right: 20%;
    }
    
    /* Input styling */
    .stChatInput {
        background-color: var(--surface-color);
        border-radius: 25px;
        border: 2px solid var(--border-color);
        padding: 1rem;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.3);
    }
    
    /* Sidebar button styling */
    .sidebar-button {
        width: 100%;
        background-color: var(--surface-color);
        border: 2px solid var(--border-color);
        border-radius: 10px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-weight: 500;
        color: var(--text-primary);
        transition: all 0.2s ease;
    }
    
    .sidebar-button:hover {
        border-color: var(--primary-color);
        background-color: rgba(30, 64, 175, 0.05);
    }
    
    /* Success/Error messages */
    .success-message {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid var(--success-color);
        padding: 1rem;
        border-radius: 8px;
        color: var(--success-color);
        font-weight: 500;
    }
    
    .error-message {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid var(--error-color);
        padding: 1rem;
        border-radius: 8px;
        color: var(--error-color);
        font-weight: 500;
    }
    
    /* Statistics cards */
    .stat-card {
        background-color: var(--surface-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stat-card h3 {
        color: var(--primary-color);
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-card p {
        color: var(--text-secondary);
        font-weight: 500;
        margin: 0.5rem 0 0 0;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 1rem;
        color: var(--text-secondary);
        font-style: italic;
    }
    
    .typing-dots {
        display: inline-flex;
        margin-left: 10px;
    }
    
    .typing-dots span {
        background-color: var(--primary-color);
        border-radius: 50%;
        width: 8px;
        height: 8px;
        margin: 0 2px;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            margin: -1rem -1rem 1rem -1rem;
        }
        
        .main-header h1 {
            font-size: 1.5rem;
        }
        
        .stChatMessage[data-testid="chat-message-user"] {
            margin-left: 10%;
        }
        
        .stChatMessage[data-testid="chat-message-assistant"] {
            margin-right: 10%;
        }
    }
</style>
""", unsafe_allow_html=True)

# App configuration
BOT_ID = "botFinance"
USER_ID = "1"
API_BASE_URL = "http://chatbot-api:8000"

class ChatApp:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(int(time.time()))
        if "total_messages" not in st.session_state:
            st.session_state.total_messages = 0
        if "current_typing" not in st.session_state:
            st.session_state.current_typing = False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=20))
    def send_user_request(self, text: str) -> Dict[str, Any]:
        """Send user message to the API"""
        url = f"{API_BASE_URL}/chat/complete"
        payload = {
            "user_message": text,
            "user_id": str(USER_ID),
            "bot_id": BOT_ID,
            "sync_request": False,
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            if response.status_code != 200:
                logger.error(f"send_user_request non-200 status: {response.status_code} body: {response.text}")
                raise requests.RequestException(f"Request failed: {response.text}")
            try:
                resp_json = response.json()
            except ValueError:
                logger.error(f"send_user_request invalid JSON response: {response.text}")
                raise requests.RequestException("Invalid JSON response from server")
            return resp_json
        except requests.RequestException as e:
            logger.error(f"Error getting response: {e}")
            raise

    def display_formatted_content(self, content: str):
        """Display content with forced formatting using container and multiple elements"""
        import re
        
        # Split content by numbered patterns
        parts = re.split(r'(\d+\.)', content)
        
        if len(parts) <= 1:
            # No numbered list found, display as is
            st.text(content)
            return
            
        # Use container for better control
        with st.container():
            # Display introduction text (if any)
            if parts[0].strip():
                st.text(parts[0].strip())
                st.text("")  # Empty line
                
            # Process numbered items
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    number = parts[i]  # "1.", "2.", etc
                    text = parts[i + 1].strip()  # corresponding text
                    
                    if text:
                        # Display each numbered item on separate lines
                        st.text(f"{number} {text}")
                        st.text("")  # Empty line for spacing

    def format_content_for_display(self, content: str) -> str:
        """Format content for proper display with forced line breaks"""
        if not content:
            return content
            
        import re
        
        # Force line breaks by adding double newlines and spaces
        # Split by numbered patterns and rejoin with forced spacing
        parts = re.split(r'(\d+\.)', content)
        
        if len(parts) <= 1:
            # No numbered list found, return as is
            return content
            
        formatted_parts = []
        for i, part in enumerate(parts):
            if re.match(r'^\d+\.$', part):  # This is a number like "1."
                if i > 0:  # Add line breaks before each numbered item (except first)
                    formatted_parts.append('\n\n')
                formatted_parts.append(f"**{part}**")
            else:
                # This is the text content, trim whitespace
                cleaned_part = part.strip()
                if cleaned_part:
                    formatted_parts.append(f" {cleaned_part}")
        
        result = ''.join(formatted_parts)
        
        # Additional formatting - ensure each numbered item is on its own line
        result = re.sub(r'(\d+\.\*\*)\s+', r'\1 ', result)
        
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=20))
    def get_bot_response(self, request_id: str) -> tuple[int, Dict[str, Any]]:
        """Get bot response from the API"""
        url = f"{API_BASE_URL}/chat/complete/{request_id}"
        
        try:
            response = requests.get(url, timeout=300)
            if response.status_code != 200:
                logger.error(f"get_bot_response non-200 status: {response.status_code} body: {response.text}")
                raise requests.RequestException(f"Get response failed: {response.text}")
            try:
                resp_json = response.json()
            except ValueError:
                logger.error(f"get_bot_response invalid JSON response: {response.text}")
                raise requests.RequestException("Invalid JSON response from server")
            return response.status_code, resp_json
        except requests.RequestException as e:
            logger.error(f"Error getting response: {e}")
            raise

    def get_chat_complete(self, text: str) -> str:
        """Complete chat interaction"""
        try:
            user_request = self.send_user_request(text)
            # Validate user_request
            if not user_request or not isinstance(user_request, dict) or "task_id" not in user_request:
                logger.error(f"Invalid user_request response: {user_request}")
                raise Exception("Server không trả về task id. Vui lòng thử lại.")

            request_id = user_request.get("task_id")
            status_code, chat_response = self.get_bot_response(request_id)

            # Validate chat_response
            if status_code != 200 or not chat_response or not isinstance(chat_response, dict):
                logger.error(f"Invalid chat_response: status={status_code}, body={chat_response}")
                raise Exception("Server không trả về kết quả hợp lệ. Vui lòng thử lại.")

            # Check if task is still processing
            task_result = chat_response.get("task_result")
            task_status = chat_response.get("task_status", "UNKNOWN")
            
            # If task is still processing, raise exception to trigger retry
            if task_status in ["PENDING", "STARTED", "RETRY"] or task_result is None:
                logger.info(f"Task still processing: status={task_status}, retrying...")
                raise requests.RequestException(f"Task still processing: {task_status}")

            # Handle different response formats
            content = None
            if isinstance(task_result, dict):
                # Format: {"role": "assistant", "content": "..."}
                content = task_result.get("content")
            elif isinstance(task_result, str):
                # Format: direct string content
                content = task_result
            
            if not content:
                logger.error(f"No content found in task_result: {task_result}")
                raise Exception("Nội dung trả về rỗng. Vui lòng thử lại.")

            # Process content to ensure proper markdown formatting
            content = self.format_content_for_display(content)
            
            logger.info("Chat response received successfully")
            return content
            
        
                
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}. Vui lòng thử lại."

    def response_generator(self, user_message: str):
        """Generate streaming response"""
        try:
            st.session_state.current_typing = True
            res = self.get_chat_complete(user_message)
            st.session_state.current_typing = False
            
            # Ensure proper line breaks and formatting are preserved
            # Convert various line break formats to markdown-friendly format
            res = res.replace('\r\n', '\n').replace('\r', '\n')
            
            # Stream the response word by word for better UX
            words = res.split()
            buffer = []
            
            for i, word in enumerate(words):
                buffer.append(word)
                # Yield every 3 words or at the end
                if len(buffer) == 3 or i == len(words) - 1:
                    yield " ".join(buffer) + " "
                    buffer = []
                    time.sleep(0.05)
            
        except Exception as e:
            st.session_state.current_typing = False
            yield f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>⚖️ Trợ Lý Pháp Lý AI</h1>
            <p class="subtitle">Hệ thống tư vấn pháp lý thông minh với công nghệ RAG</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with controls and information"""
        with st.sidebar:
            st.markdown("### 🎛️ Điều khiển")
            
            # Clear chat button
            if st.button("🗑️ Xóa cuộc trò chuyện", key="clear_chat", help="Xóa tất cả tin nhắn"):
                st.session_state.messages = []
                st.session_state.total_messages = 0
                st.session_state.conversation_id = str(int(time.time()))
                st.rerun()
            
            # Export conversation button
            if st.session_state.messages:
                if st.button("📥 Xuất cuộc trò chuyện", key="export_chat", help="Tải xuống cuộc trò chuyện"):
                    self.export_conversation()
            
            st.divider()
            
            # Conversation statistics
            st.markdown("### 📊 Thống kê")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{len(st.session_state.messages)}</h3>
                    <p>Tin nhắn</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{user_messages}</h3>
                    <p>Câu hỏi</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Legal categories (example)
            st.markdown("### ⚖️ Lĩnh vực pháp lý")
            categories = [
                "📜 Luật Dân sự",
                "🏢 Luật Doanh nghiệp", 
                "⚖️ Luật Hình sự",
                "🏠 Luật Đất đai",
                "👥 Luật Lao động",
                "📋 Luật Hành chính"
            ]
            
            for category in categories:
                if st.button(category, key=f"cat_{category}", help=f"Hỏi về {category}"):
                    sample_question = f"Tôi muốn tìm hiểu về {category.split(' ', 1)[1]}"
                    st.session_state.messages.append({"role": "user", "content": sample_question})
                    st.rerun()
            
            st.divider()
            
            # App information
            st.markdown("### ℹ️ Thông tin")
            st.info("""
            **Trợ lý pháp lý AI** sử dụng công nghệ RAG để cung cấp tư vấn pháp lý chính xác dựa trên văn bản pháp luật Việt Nam.
            
            ⚠️ **Lưu ý**: Thông tin chỉ mang tính tham khảo. Vui lòng tham khảo ý kiến chuyên gia cho các vấn đề phức tạp.
            """)

    def export_conversation(self):
        """Export conversation to downloadable file"""
        if not st.session_state.messages:
            st.warning("Không có cuộc trò chuyện nào để xuất.")
            return
        
        # Create conversation data
        conversation_data = {
            "conversation_id": st.session_state.conversation_id,
            "export_time": datetime.datetime.now().isoformat(),
            "total_messages": len(st.session_state.messages),
            "messages": []
        }
        
        for i, msg in enumerate(st.session_state.messages):
            conversation_data["messages"].append({
                "index": i + 1,
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp", "Unknown")
            })
        
        # Convert to JSON
        json_str = json.dumps(conversation_data, ensure_ascii=False, indent=2)
        
        # Create download button
        st.download_button(
            label="📥 Tải xuống (JSON)",
            data=json_str,
            file_name=f"legal_chat_{st.session_state.conversation_id}.json",
            mime="application/json",
            key="download_json"
        )

    def render_typing_indicator(self):
        """Render typing indicator"""
        if st.session_state.current_typing:
            st.markdown("""
            <div class="typing-indicator">
                Trợ lý đang trả lời...
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def render_welcome_message(self):
        """Render welcome message when no conversations exist"""
        if not st.session_state.messages:
            st.markdown("""
            <div class="chat-container">
                <h3>👋 Chào mừng bạn đến với Trợ lý Pháp lý AI!</h3>
                <p>Tôi có thể giúp bạn:</p>
                <ul>
                    <li>🔍 Tra cứu các văn bản pháp luật Việt Nam</li>
                    <li>💡 Giải đáp các thắc mắc pháp lý</li>
                    <li>📋 Hướng dẫn các thủ tục hành chính</li>
                    <li>⚖️ Tư vấn về quyền và nghĩa vụ pháp lý</li>
                </ul>
                <p><strong>Hãy bắt đầu bằng cách đặt câu hỏi của bạn!</strong></p>
            </div>
            """, unsafe_allow_html=True)

    def run(self):
        """Main app runner"""
        self.render_header()
        self.render_sidebar()
        
        # Main chat area
        self.render_welcome_message()
        
        # Display chat messages from history
        for i, message in enumerate(st.session_state.messages):
            timestamp = message.get("timestamp", "")
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    # Use formatted display for assistant messages
                    self.display_formatted_content(message["content"])
                else:
                    # Regular display for user messages
                    st.write(message["content"])
                if timestamp:
                    st.caption(f"🕐 {timestamp}")
        
        # Show typing indicator
        self.render_typing_indicator()
        
        # Accept user input
        if prompt := st.chat_input("Đặt câu hỏi pháp lý của bạn...", key="user_input"):
            # Add timestamp to message
            timestamp = datetime.datetime.now().strftime("%H:%M - %d/%m/%Y")
            
            # Add user message to chat history
            user_message = {
                "role": "user", 
                "content": prompt,
                "timestamp": timestamp
            }
            st.session_state.messages.append(user_message)
            st.session_state.total_messages += 1
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
                st.caption(f"🕐 {timestamp}")
            
            # Display assistant response
            with st.chat_message("assistant"):
                try:
                    # Get complete response first
                    complete_response = self.get_chat_complete(prompt)
                    # Split and display each numbered item separately
                    self.display_formatted_content(complete_response)
                    response_timestamp = datetime.datetime.now().strftime("%H:%M - %d/%m/%Y")
                    st.caption(f"🕐 {response_timestamp}")
                    
                    # Add assistant response to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": complete_response,
                        "timestamp": response_timestamp
                    }
                    st.session_state.messages.append(assistant_message)
                    st.session_state.total_messages += 1
                    
                except Exception as e:
                    error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                    st.error(error_msg)
                    
                    # Add error message to history
                    error_timestamp = datetime.datetime.now().strftime("%H:%M - %d/%m/%Y")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": error_timestamp
                    })

# Run the app
if __name__ == "__main__":
    app = ChatApp()
    app.run()