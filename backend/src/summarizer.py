from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# load the model
summarizer_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def summarize_text(text):
    # prepare template for prompt
    template = """You are a very good assistant that summarizes text.

    Always keep important key points in the summary.

    ==================
    {text}
    ==================

    Write a summary of the content in Vietnamese.
    """

    prompt = template.format(text=text)

    messages = [HumanMessage(content=prompt)]
    summary = summarizer_model(messages)
    return summary.content
