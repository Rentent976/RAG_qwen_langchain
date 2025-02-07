from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_chat_chain(llm):
    # 提示模板中添加 chat_history
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个中国古诗词专家，能准确的一字不差的背诵很多古诗词，请用你最大的能力来回答用户的问题。",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    
    chat_chain = prompt | llm | StrOutputParser()
    
    return chat_chain