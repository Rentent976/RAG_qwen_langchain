# @Author：青松
# 公众号：FasterAI
# Python, version 3.10.14
# Pytorch, version 2.3.0
# Chainlit, version 1.1.301

import chainlit as cl
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 存储对话历史
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


async def send_welcome_msg():

    await cl.Message(
        content="你好！",
    ).send()


async def response_with_history_by_astream(message: cl.Message, chain, session_id):
    # 用 RunnableWithMessageHistory 包装 Chain 添加对话历史能力
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    msg = cl.Message(content="")

    # 配置中使用 session_id 进行大模型交互
    async for chunk in runnable_with_history.astream(
            {"question": message.content},
            config=RunnableConfig(configurable={"session_id": session_id},
                                  callbacks=[cl.LangchainCallbackHandler()])
    ):
        await msg.stream_token(chunk)

    await msg.send()
