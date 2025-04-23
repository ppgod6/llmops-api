from langchain_core.chat_history import InMemoryChatMessageHistory

chat_history = InMemoryChatMessageHistory()

chat_history.add_user_message("你好，我是马卡，你是谁？")
chat_history.add_ai_message("你好，我是deepseek")

print(chat_history)
