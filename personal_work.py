import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

os.environ["OpenAI_API_KEY"] = os.getenv("GPT_API")

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader("data/LLM_Research_Trends.pdf")

# 페이지 별 문서 로드
docs = loader.load()



text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)


recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)


text_splits = text_splitter.split_documents(docs)


for i, chunk in enumerate(text_splits[:1]):  # 슬라이싱으로 상위 3개만 가져옴
    print(f"Chunk {i}: {chunk}")