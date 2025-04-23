from langchain.memory.chat_memory import BaseChatMemory

memory = BaseChatMemory(
    input_key="query",
    output_key="output",
    return_messages=True,

    # chat_history
)

memory_variable = memory.load_memory_variables({})
