
import streamlit as st

from config import keys
from utils.utils import extract_text_from_PDF, split_content_into_chunks
from utils.utils import save_chunks_into_vectorstore, get_chat_chain, process_user_input
from models.llm_model import get_openaiEmbedding_model, get_embedding_model
import os

# os.environ["LANGCHAIN_TRACING_V2"]= "true"
# os.environ["LANGCHAIN_ENDPOINT"]= "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"]="langchain02"

def main():
    # 配置界面
    st.set_page_config(page_title="企业级通用知识库",
                       page_icon=":robot:")

    st.header("企业级通用知识库")

    # 初始化
    # session_state是Streamlit提供的用于存储会话状态的功能
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # 1. 提供用户输入文本框
    user_input = st.text_input("请输入你的提问: ")
    # 处理用户输入，并返回响应结果
    if user_input:
        process_user_input(user_input)

    with st.sidebar:
        # 2. 设置子标题
        st.subheader("知识文档")
        # 3. 上传文档
        files = st.file_uploader("上传知识文档，然后点击'提交并处理'",
                                 accept_multiple_files=True)
        if st.button("提交"):
            with st.spinner("请等待，处理中..."):
                # 4. 获取文档内容（文本）
                texts = extract_text_from_PDF(files)
                # 5. 将获取到的文档内容进行切分
                content_chunks = split_content_into_chunks(texts)
                # 6. 向量化并且存储数据
                # embedding_model = get_openaiEmbedding_model()
                embedding_model = get_embedding_model(keys.Keys.EMBEDDING_PATH)
                vector_store = save_chunks_into_vectorstore(content_chunks, embedding_model)
                # 7. 创建对话chain
                st.session_state.conversation = get_chat_chain(vector_store)


if __name__ == "__main__":
    main()