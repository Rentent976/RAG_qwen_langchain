import chainlit as cl
from langchain_community.llms import Tongyi
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chain_util
import msg_util
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain.vectorstores import Chroma
from langchain.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
import os

# 加载大模型
llm = Tongyi(model_name='qwen-plus')

# 加载 embedding创建模型
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个切片的字符数
    chunk_overlap=50,  # 切片之间的重叠字符数
    separators=["\n\n", "\n", "。", " ","，"],  # 切分依据
)

# 存储所有切片的列表
all_chunks = []

# 遍历 news 目录下的所有文件
for filename in os.listdir('../news'):
    if filename.endswith('.md'):
        file_path = os.path.join('../news', filename) 

        with open(file_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()
        
        # 输出文件字符数
        char_count = len(markdown_content)
        #print(f"文件: {filename}，字符数: {char_count}")

        # 将 Markdown 文本切片
        chunks = text_splitter.split_text(markdown_content)
        all_chunks.extend(chunks)

client = QdrantClient(":memory:")
# if not exists then create collection
if not client.collection_exists("rag_collection"):
    # create collection
    client.create_collection(
        "rag_collection",
        vectors_config=VectorParams(
            size=len(embeddings.embed_query("hello world")), # size:1536
            distance=Distance.COSINE
        )
    )
vector_store = QdrantVectorStore(client=client, collection_name="rag_collection", embedding=embeddings) # 初始化向量数据库
vector_store.add_texts(all_chunks) # 将切分好的文本嵌入到向量数据库中 (每次运行即将数据加入一遍)

# 管理聊天历史
store = {}

def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@cl.on_chat_start
async def on_chat_start():
    """ 监听会话开始事件 """
    
    # 添加 session_id
    session_id = "abc123"
    cl.user_session.set('session_id', session_id)

    # 发送欢迎信息
    await msg_util.send_welcome_msg()

    # 创建向量存储
    #vectorstore = await cl.make_async(Chroma.from_texts)(texts, embeddings)
    # 将 Chroma 向量数据库转化为检索器
    retriever = vector_store.as_retriever()

    # contextualize question
    question_prompt = get_contextualize_question_prompt()
    # 改写链：结合上下文改写用户问题
    history_aware_retriever = create_history_aware_retriever(llm, retriever, question_prompt)

    # qa chain
    qa_prompt_template = get_answer_prompt()
    # 问答链：根据问题和参考内容生成答案
    qa_chain = create_stuff_documents_chain(llm, qa_prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
                                    rag_chain,
                                    get_session_history,
                                    input_messages_key="input",
                                    history_messages_key="chat_history",
                                    output_messages_key="answer"
                                )
    
    # 在 rag_chain 中添加 chat_history
    cl.user_session.set("conversational_rag_chain", conversational_rag_chain)

    # 初始化链
    #init_chains()


@cl.on_message
async def on_message(message: cl.Message):
    """ 监听用户消息事件 """
    session_id = cl.user_session.get("session_id")
    # 获得对话链
    conversational_rag_chain = cl.user_session.get("conversational_rag_chain")

    msg = cl.Message(content="")

    # 使用 session_id 用流式的方式响应用户问题
    chain = conversational_rag_chain.pick("answer")  # 只挑选 'answer' 属性输出
    async for chunk in chain.astream(
            {"input": message.content},
            config=RunnableConfig(
                configurable={"session_id": session_id},
                callbacks=[cl.LangchainCallbackHandler()])
    ):
        await msg.stream_token(chunk)

    await msg.send()


def init_chains():
    """ 初始化系统中的链 """
    # 对话链
    chat_chain = chain_util.get_chat_chain(llm)
    cl.user_session.set("chat_chain", chat_chain)

# 改写问题prompt
def get_contextualize_question_prompt():
    """
    基于历史记录来改写用户问的问题
    :return:
    """
    system_prompt = """\
    请根据聊天历史和最后用户的问题，改写用户最终提出的问题。
    你只需要改写用户最终的问题，请不要回答问题
    没有聊天历史则将用户问题直接返回，有聊天历史则进行改写
    """
    contextualize_question_prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return contextualize_question_prompt

# 提问prompt
def get_answer_prompt():
    system_prompt = """\
    你是一个问答任务的助手，请依据以下检索出来的信息去回答问题，回答的字数控制在100字内：
    {context}
    """
    qa_prompt = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return qa_prompt

### 启动服务
# chainlit run app.py -w 
