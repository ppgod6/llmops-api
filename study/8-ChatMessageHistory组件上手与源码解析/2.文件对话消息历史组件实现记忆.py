import dotenv
from langchain_community.chat_message_histories import FileChatMessageHistory
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-12d9e5db3b2b4bc49119a0dadff2d5f9")
chat_history = FileChatMessageHistory("./memory.txt")
while True:

    query = input("Human:")

    if query == "q":
        exit(0)

    print("AI:", flush=True, end="")
    system_prompt = (
        "你是deepseek聊天机器人，可以根据相应的上下文回复用户信息，上下文里存放的是人类与你对话的信息列表。\n\n"
        f"<context>{chat_history}</context>\n\n"
    )
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    ai_content = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is None:
            break
        ai_content += content
        print(content, flush=True, end="")
    chat_history.add_user_message(query)
    chat_history.add_ai_message(ai_content)
    print("")
