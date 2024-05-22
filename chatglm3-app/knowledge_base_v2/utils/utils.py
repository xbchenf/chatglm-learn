
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS#, Milvus, Pinecone, Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from config import keys
from models.llm_model import get_openai_model, get_chatglm3_6b_model
#import pinecone
import streamlit as st
from PyPDF2 import PdfReader

from prompt.prompt_templates import bot_template, user_template
import  json
import re


"""
可以扩展支持多个文件格式
"""
def extract_text_from_PDF(files):
    text = ""
    for pdf in files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
"""
 核心优化地方
"""
def split_content_into_chunks(text):
    text_spliter = CharacterTextSplitter(separator="\n",
                                         chunk_size=50,
                                         chunk_overlap=5,
                                         length_function=len)
    chunks = text_spliter.split_text(text)
    return chunks
"""
数据库选型
"""
def save_chunks_into_vectorstore(content_chunks, embedding_model):
    # ① FAISS
    # pip install faiss-gpu (如果没有GPU，那么 pip install faiss-cpu)
    vectorstore = FAISS.from_texts(texts=content_chunks,
                                      embedding=embedding_model)
    # ② Pinecone
    # pip install pinecone-client==2.2.2
    # 初始化
    # pinecone.init(api_key=Keys.PINECONE_KEY, environment="asia-southeast1-gcp")
    # # 创建索引
    # index_name = "pinecone-chatbot-demo"
    # # 检查索引是否存在，如果不存在，则创建
    # if index_name not in pinecone.list_indexes():
    #     pinecone.create_index(name=index_name,
    #                           metric="cosine",
    #                           dimension=1536)
    # vectorstore = Pinecone.from_texts(texts=content_chunks,
    #                                       embedding=embedding_model,
    #                                       index_name=index_name)

    # ③ Milvus, pip install pymilvus
    # 要么安装到云上
    # 要么安装到虚拟机上
    # vectorstore = Milvus.from_texts(texts=content_chunks,
    #                                     embedding=embedding_model,
    #                                     connection_args={"host": "localhost", "port": "19530"},
    # )

    return vectorstore
"""
核心优化点
"""
def get_chat_chain(vector_store):
    # ① 获取 LLM model（核心优化地方）
    #llm = get_openai_model()
    llm=get_chatglm3_6b_model(keys.Keys.CHATGLM3_MODEL_PATH)
    #llm = get_huggingfacehub(model_name="google/flan-t5-xxl")

    # 用于缓存或者保存对话历史记录的对象
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    # ③ 对话链
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_data(str):
    # 清除空格
    r1 = str.strip()
    # 清楚无用的信息
    r2 = r1.replace("Action: ", "").replace("```", "")
    # 转化为json格式
    r3 = json.loads(r2)
    # 返回信息
    return r3["action_input"]

def process_user_input(user_input):
    if st.session_state.conversation is not None:
        # 调用函数st.session_state.conversation，并把用户输入的内容作为一个问题传入，返回响应。
        print("用户输入问题："+user_input)
        response = st.session_state.conversation({'question': user_input})
        # session状态是Streamlit中的一个特性，允许在用户的多个请求之间保存数据。
        st.session_state.chat_history = response['chat_history']
        # 显示聊天记录
        # chat_history : 一个包含之前聊天记录的列表
        for i, message in enumerate(st.session_state.chat_history):
            # 用户输入
            if i % 2 == 0:
                print("用户输入：")
                print(message.content)
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True) # unsafe_allow_html=True表示允许HTML内容被渲染
            else:
                # 机器人响应
                print("机器人回应：")
                message_result=get_data(str(message.content))
                print(message_result)
                st.write(bot_template.replace(
                    "{{MSG}}", message_result), unsafe_allow_html=True)
