from typing import Any

import dotenv
from openai import OpenAI

dotenv.load_dotenv()


class ConversationSummaryBufferMemory:
    def __init__(self, summary: str = "", chat_histories: list = None, max_tokens: int = 300):
        self.summary = summary
        self.chat_histories = [] if chat_histories is None else chat_histories
        self.max_tokens = max_tokens
        self._client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-12d9e5db3b2b4bc49119a0dadff2d5f9")

    @classmethod
    def get_num_tokens(self, query: str) -> int:
        return len(query)

    def save_context(self, human_query: str, ai_content: str) -> None:
        self.chat_histories.append({"human": human_query, "ai": ai_content})

        buffer_string = self.get_buffer_string()

        tokens = self.get_num_tokens(buffer_string)

        if tokens > self.max_tokens:
            first_chat = self.chat_histories[0]
            print("新摘要生成中...")
            self.summary = self.summary_text(
                self.summary,
                f"Human:{first_chat.get('human')}\nAI:{first_chat.get('ai')}"
            )
            print("新摘要生成成功：", self.summary)
            del self.chat_histories[0]

    def get_buffer_string(self) -> str:
        buffer: str = ""
        for chat in self.chat_histories:
            buffer += f"Human:{chat.get('human')}\nAI:{chat.get('ai')}\n\n"
        return buffer.strip()

    def load_memory_variables(self) -> dict[str, Any]:
        buffer_string = self.get_buffer_string()
        return {
            "chat_history": f"摘要:{self.summary}\n\n历史信息:{buffer_string}\n"
        }

    def summary_text(self, origin_summary: str, new_line: str) -> str:
        prompt = f"""你是一个强大的聊天机器人，请根据用户提供的谈话内容，总结摘要，并将其添加到先前提供的摘要中，返回一个新的摘要，除了新摘要其他任何数据都不要生成，如果用户的对话信息里有一些关键的信息，比方说姓名、爱好、性别、重要事件等等，这些全部都要包括在生成的摘要中，摘要尽可能要还原用户的对话记录。

请不要将&lt;example&gt;标签里的数据当成实际的数据，这里的数据只是一个示例数据，告诉你该如何生成新摘要。

&lt;example&gt;
当前摘要：人类会问人工智能对人工智能的看法，人工智能认为人工智能是一股向善的力量。

新的对话：
Human：为什么你认为人工智能是一股向善的力量？
AI：因为人工智能会帮助人类充分发挥潜力。

新摘要：人类会问人工智能对人工智能的看法，人工智能认为人工智能是一股向善的力量，因为它将帮助人类充分发挥潜力。
&lt;/example&gt;

=====================以下的数据是实际需要处理的数据=====================

当前摘要：{origin_summary}

新的对话：
{new_line}

请帮用户将上面的信息生成新摘要。"""
        completion = self._client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return completion.choices[0].message.content


client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-12d9e5db3b2b4bc49119a0dadff2d5f9")
memory = ConversationSummaryBufferMemory("", [], 300)
while True:
    query = input('Human:')

    if query == 'q':
        break

    memory_variables = memory.load_memory_variables()
    answer_prompt = (
        "你是一个强大的聊天机器人，请根据对应的上下文和用户提问解决问题。\n\n"
        f"{memory_variables.get('chat_history')} \n\n"
        f"用户的提问是：{query}"
    )
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=[
            {"role": "user", "content": answer_prompt},
        ],
        stream=True,
    )

    print("AI:", flush=True, end="")
    ai_content = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is None:
            break
        ai_content += content
        print(content, flush=True, end="")
    print("")
    memory.save_context(query, ai_content)
