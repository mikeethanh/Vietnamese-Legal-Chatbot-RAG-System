import asyncio
import logging
from xml.dom import ValidationErr

from cache import get_conversation_id
from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from utils import setup_logging

from database import engine, get_db

Base = declarative_base()
Base.metadata.create_all(bind=engine)
db = next(get_db())
setup_logging()
logger = logging.getLogger(__name__)


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(50), nullable=False, default="")
    bot_id = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=False)
    message = Column(String)  # Assuming TextField is equivalent to String in SQLAlchemy
    is_request = Column(Boolean, default=True)
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )


def load_conversation(conversation_id: str):
    return (
        db.query(ChatConversation)
        .filter(ChatConversation.conversation_id == conversation_id)
        .order_by(ChatConversation.created_at)
        .all()
    )


async def read_conversation(conversation_id: str):
    async with db() as session:
        result = await session.execute(
            select(ChatConversation).where(
                ChatConversation.conversation_id == conversation_id
            )
        )
        db_conversation = result.scalars().first()
        if db_conversation is None:
            raise ValidationErr("Conversation not found")
        return db_conversation


def convert_conversation_to_openai_messages(user_conversations):
    conversation_list = [
        {"role": "system", "content": "You are an amazing virtual assistant"}
    ]

    for conversation in user_conversations:
        role = "assistant" if not conversation.is_request else "user"
        content = str(conversation.message)
        conversation_list.append({"role": role, "content": content})

    logging.info(f"Create conversation to {conversation_list}")

    return conversation_list


def update_chat_conversation(
    bot_id: str, user_id: str, message: str, is_request: bool = True
):
    # Step 1: Create a new ChatConversation instance
    conversation_id = get_conversation_id(bot_id, user_id)

    new_conversation = ChatConversation(
        conversation_id=conversation_id,
        bot_id=bot_id,
        user_id=user_id,
        message=message,
        is_request=is_request,
        completed=not is_request,
    )
    # Step 4: Save the ChatConversation instance
    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)

    logger.info(f"Create message for conversation {conversation_id}")

    return conversation_id


def get_conversation_messages(conversation_id):
    user_conversations = load_conversation(conversation_id)
    return convert_conversation_to_openai_messages(user_conversations)


class Document(Base):
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String(2000), nullable=False, default="")
    content = Column(String)  # Assuming TextField is equivalent to String in SQLAlchemy
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )


# insert document into database
def insert_document(question: str, content: str):
    # Step 1: Create a new Document instance
    new_doc = Document(
        question=question,
        content=content,
    )
    # Step 2: Save the Document instance
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    logger.info(f"Create document successfully {new_doc}")

    return new_doc
