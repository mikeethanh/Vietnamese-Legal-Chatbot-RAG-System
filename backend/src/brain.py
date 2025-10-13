import json
import logging
import os
from openai import OpenAI
from redis import InvalidResponse

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", default=None)

def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

client = get_openai_client()

def openai_chat_complete(messages=(), model="gpt-4o-mini", raw=False):
    logger.info("Chat complete for {}".format(messages))
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    if raw:
        return response.choices[0].message
    output = response.choices[0].message
    logger.info("Chat complete output: ".format(output))
    return output.content

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def gen_doc_prompt(docs):
    """
    Document:
    Title: Uong atiso ...
    Content: ....
    """
    doc_prompt = ""
    for doc in docs:
        doc_prompt += f"Title: {doc['title']} \n Content: {doc['content']} \n"

    return "Document: \n + {}".format(doc_prompt)


def generate_conversation_text(conversations):
    conversation_text = ""
    for conversation in conversations:
        logger.info("Generate conversation: {}".format(conversation))
        role = conversation.get("role", "user")
        content = conversation.get("content", "")
        conversation_text += f"{role}: {content}\n"
    return conversation_text


def detect_user_intent(history, message):
    # Convert history to list messages
    history_messages = generate_conversation_text(history)
    logger.info(f"History messages: {history_messages}")
    # Update documents to prompt
    user_prompt = f"""
    Given following conversation and follow up question, rephrase the follow up question to a standalone question in Vietnamese.

    Chat History:
    {history_messages}

    Original Question: {message}

    Answer:
    """
    openai_messages = [
        {"role": "system", "content": "You are an amazing virtual assistant"},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Rephrase input messages: {openai_messages}")
    # call openai
    return openai_chat_complete(openai_messages)



def detect_route(history, message):
    logger.info(f"Detect route on history messages: {history}")
    # Update documents to prompt
    user_prompt = f"""
    Given the following chat history and the user's latest message, determine whether the user's intent is to ask for a frequently asked question ("bank_faq") or to inquire about loans and savings ("loan_savings") or to play some games ("play_game"). \n
    Provide only the classification label as your response.

    Chat History:
    {history}

    Latest User Message:
    {message}

    Classification (choose either "bank_faq" or "loan_savings" or "play_game"):
    """
    openai_messages = [
        {"role": "system", "content": "You are a highly intelligent assistant that helps classify customer queries"},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Route output: {openai_messages}")
    # call openai
    return openai_chat_complete(openai_messages)


def get_financial_tools():
    tools = []
    logger.info(f"Financial tools: {tools}")
    return tools


def get_financial_agent_answer(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()

    # Execute the chat completion request
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )

    # Attempt to extract response details
    if not resp.choices:
        logger.error("No choices available in the response.")
        return {"role": "assistant", "content": "An error occurred, please try again later."}

    choice = resp.choices[0]
    return choice


def convert_tool_calls_to_json(tool_calls):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "arguments": json.dumps(call.function.arguments),
                    "name": call.function.name
                }
            }
            for call in tool_calls
        ]
    }

def get_financial_agent_handle(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()
    choice = get_financial_agent_answer(messages, model, tools)

    resp_content = choice.message.content
    resp_tool_calls = choice.message.tool_calls
    # Prepare the assistant's message
    if resp_content:
        return resp_content

    elif resp_tool_calls:
        logger.info(f"Process the tools call: {resp_tool_calls}")
        # List to hold tool response messages
        tool_messages = []
        # Iterate through each tool call and execute the corresponding function
        for tool_call in resp_tool_calls:
            # Display the tool call details
            logger.info(f"Tool call: {tool_call.function.name}({tool_call.function.arguments})")
            # Retrieve the tool function from available tools
            tool = available_tools[tool_call.function.name]
            # Parse the arguments for the tool function
            tool_args = json.loads(tool_call.function.arguments)
            # Execute the tool function and get the result
            result = tool(**tool_args)
            tool_args['result'] = result
            # Append the tool's response to the tool_messages list
            tool_messages.append({
                "role": "tool",  # Indicate this message is from a tool
                "content": json.dumps(tool_args),  # The result of the tool function
                "tool_call_id": tool_call.id,  # The ID of the tool call
            })
        # Update the new message to get response from LLM
        # Append the tool messages to the existing messages
        # Check here: https://platform.openai.com/docs/guides/function-calling
        next_messages = messages + [convert_tool_calls_to_json(resp_tool_calls)] + tool_messages
        return get_financial_agent_handle(next_messages, model, tools)
    else:
        raise InvalidResponse(f"The response is invalid: {choice}")
