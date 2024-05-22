#!/usr/bin/env python
from PyPDF2 import PdfReader
from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda

from config import keys
from models.llm_model import get_openaiEmbedding_model, get_openai_model
from prompt.prompt_templates import user_template, bot_template
from utils.utils import extract_text_from_PDF, split_content_into_chunks, save_chunks_into_vectorstore

from langserve import add_routes
from dotenv import load_dotenv



load_dotenv()
##FastAPI是一个基于Python的Web框架，用于构建高性能、可扩展的API。它提供了一种简单、直观的方式来定义API端点，以及处理HTTP请求和响应。
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)


def getContext(file_path):
    text = ""
    print("文件位置："+file_path)
    pdf_reader = PdfReader(file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

texts = getContext("../data/本地知识库.pdf")
content_chunks = split_content_into_chunks(texts)
embedding_model = get_openaiEmbedding_model()
vector_store = save_chunks_into_vectorstore(content_chunks, embedding_model)

llm = get_openai_model()

memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

def get_knowledge_by_input(input_str:str):

    print("输入内容："+input_str)
    result = conversation_chain({'question': input_str})
    print(result)
    return result



"""
开放接口
"""
add_routes(app,
           RunnableLambda(get_knowledge_by_input),
           path="/get_knowledge"
           )


if __name__ == "__main__":
    import uvicorn
    ## Python的web服务器
    uvicorn.run(app, host="localhost", port=9999)