from langchain_core.prompts import ChatPromptTemplate

system_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是OpenAI开发的聊天机器人，请根据用户的提问进行回复，我叫{username}"),
])

human_chat_prompt = ChatPromptTemplate.from_messages([
    ("human", "{query}")
])

chat_prompt = system_chat_prompt + human_chat_prompt

print(chat_prompt.invoke({
    "username": "硬汉马卡",
    "query": "你好,你是？"
}))
