# **개인 프로젝트 파일**
## **1. 사용환경 준비**
### **1.1. 패키지**

프로그램에 필요한 패키지와 클래스들을 모두 불러오는 단계이다.

# 환경 변수를 불러오기 위한 os 모듈을 사용한다.
import os

# re는 정규 표현식(Regular Expression) 관련 작업을 처리하는 모듈이다.
# re.sub는 특정 패턴에 해당하는 문자열을 다른 문자열로 치환하거나 삭제하는 기능을 제공한다.
import re

# FAISS(Facebook AI Similarity Search)는 벡터 검색과 유사도 매칭을 위한 라이브러리이다.
import faiss

# similarity score 평균 계산을 위해 Python 내장 통계 모듈의 mean 함수를 사용한다.
from statistics import mean

# LangChain 패키지에서 OpenAI 기반의 채팅 모델을 활용하기 위한 ChatOpenAI 클래스를 불러온다.
from langchain_openai import ChatOpenAI

# LangChain 패키지에서 OpenAI 임베딩을 생성하기 위한 OpenAIEmbeddings 클래스를 불러온다.
from langchain_openai import OpenAIEmbeddings

# LangChain 패키지에서 텍스트를 분리하기 위한 도구인 CharacterTextSplitter와 RecursiveCharacterTextSplitter를 불러온다.
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# LangChain 커뮤니티 패키지에서 PDF 파일을 로드하기 위한 PyPDFLoader 클래스를 불러온다.
from langchain_community.document_loaders import PyPDFLoader

# 벡터 데이터베이스를 생성하기 위해 FAISS 클래스와 InMemoryDocstore 클래스를 불러온다.
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# LangChain을 이용하여 RAG(Retrieval-Augmented Generation) 체인을 생성하는 데 필요한 ChatPromptTemplate 및 RunnablePassthrough 클래스를 불러온다.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 벡터 데이터베이스에 텍스트를 추가하기 위해 Document 클래스를 불러온다.
from langchain_core.documents import Document

# 고유한 ID(UUID)를 생성하기 위해 uuid4 함수를 사용한다.
from uuid import uuid4

# 도전과제에서의 프린트 시간을 생성하기 위한 기능 불러오기.
from datetime import datetime

---
## **2. API 환경변수 설정 및 모델 초기화**

API 키를 설정하고 모델의 초기화를 하는 단계이다.




# OpenAI_API_KEY에 현재 나의 GPT_API키의 값을 가져와서 할당한다.
os.environ["OpenAI_API_KEY"] = os.getenv("GPT_API")

# gpt-4o 모델의 초기화.
model = ChatOpenAI(model ="gpt-4o")
---
## **3. PDF 파일 로드 및 모델 초기화**

PDF 파일을 로드하고, 페이지 별로 문서를 로드하는 단계이다.
# PDF 파일 로드. 파일의 경로 입력 
# pdf는 과제 파일/ pdf1은 새로 추가한 파일이다.
pdf = PyPDFLoader("data/LLM_Research_Trends.pdf")
pdf1 = PyPDFLoader("data/Research_Trends_in_LLM_and_Mathematical_Reasoning.pdf")

# 페이지 별 문서 로드
pdf_loader = pdf.load()
pdf1_loader = pdf1.load()
---
### **3.1. 데이터 파악 밎 천처리**

PDF 파일을 살펴보면, 페이지 내에 필요 없는 단어나 꾸밈 요소가 많이 포함되어 있다.  
  
이러한 불필요한 요소들은 학습의 효율성을 저하시킬 수 있으므로,  
데이터 전처리를 통해 제거한다.

---

#### **전처리 대상**

1. **각주와 주석**  
   - 본문에 불필요한 추가 정보를 포함하는 각주 및 주석.

2. **각종 URL**  
   - 텍스트 내 포함된 웹사이트 주소 및 링크.

3. **각종 첨부 문자**  
   - 파일 내 특정 문자나 기호.

4. **PDF 특성으로 인한 줄넘김**  
   - PDF 형식으로 인해 생기는 부자연스러운 줄바꿈.

5. **중복된 공백**  
   - 연속된 공백으로 인해 텍스트가 불필요하게 길어지는 문제.

6. **레퍼런스**  
   - 학술적 참고자료 목록.

---

#### **추가 전처리 사항**
- 새로운 파일 추가와 함께, 특정 문자의 제거 작업을 추가로 진행.

---

### **데이터 전처리 방법**
- 정규식을 이용한 데이터의 분별 후 제거및 다른 데이터로 교체 할 예정.

# 줄바꿈 제거 및 각주/참고문헌 제거 함수
def clean_text(documents):
    for doc in documents:
        content = doc.page_content
         
        # 각주 제거: [숫자], [숫자, 숫자], [숫자-숫자] 형태
        content = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", content)
        
        # [숫자 숫자 , 숫자 숫자] 또는 [숫자, 숫자] 패턴 제거
        content = re.sub(r"\[\s*(\d+\s*,?\s*)+\]", "", content)
        
        # 특정 패턴 제거: 1), 23), 등
        content = re.sub(r"\d+\)", "", content)
        
        
        # 링크 제거: http://, https://, www. 등으로 시작하는 URL
        content = re.sub(r"https?://\S+|www\.\S+", "", content)

        # 불필요한 줄바꿈 제거 (문장 끝은 유지)
        content = re.sub(r"(?<![.!?])\n", "", content)

        # 중복 공백 제거
        content = re.sub(r"\s{2,}", " ", content)
        

        # 특정 텍스트의 제거
        content = re.sub(r"참고문헌.*", "", content, flags=re.DOTALL) # 참고문헌과 그 이후의 데이터 전부 제거     
        content = re.sub(r"2023. 11 정보과학회지.\d+\s", "", content) # 특정 문구과 그 뒤의 숫자 제거
        content = re.sub(r"\d+\s*특집원고 초거대 언어모델 연구 동향", "", content) # 특정 문구와 그 앞의 숫자 제거
        content = re.sub(r"\d+\s*권오욱 외 / 초거대 언어모델과 수학추론 연구 동향", "", content)
        content = re.sub(r"특집원고","",content)
        
        
        # 정리된 텍스트 저장
        doc.page_content = content.strip()

    return documents
---
### **3.2. 전처리 과정을 거친 데이터**

전처리 함수를 통해 PDF 파일의 데이터를 정제합니다.  
정제된 데이터는 `cleaned_doc` 객체에 저장되며, 이후의 추가 처리 작업에 활용됩니다.

---

#### **전처리 과정**
1. PDF 파일을 전처리 함수(`clean_text`)에 입력합니다.
2. 함수는 필요 없는 각주, URL, 중복 공백, 특정 패턴 등을 제거합니다.
3. 전처리된 데이터를 `cleaned_doc`에 저장합니다.

cleaned_pdf = clean_text(pdf_loader)
cleaned_pdf1 = clean_text(pdf1_loader)
---
## **4. 문서 청크로 나누기**

### **4.1 CharacterTextSplitter**

`CharacterTextSplitter`는 고정된 기준을 가지고 텍스트를 분할하는 도구입니다.  
아래는 각 주요 매개변수와 그 동작에 대한 설명입니다.

---

**1. `separator`**
- 텍스트를 분할할 때 사용하는 기준을 지정합니다.
- 예: `"\n\n"`은 줄바꿈이 두 번 일어난 부분을 기준으로 텍스트를 분리합니다.  
  - 이는 문단 단위의 분할을 의미합니다.
- PDF의 특성상 `"\n"`은 문장마다 줄바꿈이 되어 있고, `"\n\n"`은 대체로 한 페이지를 의미합니다.

---

 **2. `chunk_size`**
- 분할된 각 조각의 **최대 길이**를 설정합니다.
- 이상하게 동작하는 경우가 있다.
  - 조각의 길이가 100을 넘는데도 허용되거나,
  - `"\n"` 기준으로 문장이 분리되어 각 조각에 단어가 적음에도 불구하고 `chunk_size` 초과 에러가 발생.
- 이로 인해 **각 조각의 최대 길이**라기보다는, **총 조각의 최대 개수**로 작동하는 것이 아닌가 추측됩니다.

---

 **3. `chunk_overlap`**
- 분할된 조각 간의 **중복 길이**를 설정합니다.
- 문제점:
  - 페이지마다 분할이 이루어지는 경우, 앞뒤 내용이 짤리지만 `overlap`이 적용되지 않음.
  - 이는 데이터를 **페이지 단위로 분할**한 방식 때문으로 보입니다.

---

 **4. `length_function`**
- 텍스트의 길이를 측정하는 방식을 설정합니다.
- 기본값은 Python의 `len()` 함수로, 문자열의 길이를 기준으로 측정합니다.
- 필요에 따라 다른 함수(예: 단어 수 계산)를 정의해 사용할 수 있습니다.

---

 **5. `is_separator_regex`**
- `separator`가 정규식(Regular Expression)으로 처리될 수 있는지 여부를 설정합니다.
- `True`로 설정하면 `separator`에 정규식을 사용하여 텍스트를 분리할 수 있습니다.
  - 예: `separator=r"\d+\."`은 숫자와 점(`.`)을 기준으로 분리합니다.

---

 **요약**
`CharacterTextSplitter`는 텍스트 분할을 위한 강력한 도구지만, PDF와 같은 특정 형식에서는 예상과 다르게 동작할 수 있습니다. 이는 PDF의 줄바꿈 구조나 데이터 로드 방식의 특수성 때문일 가능성이 큽니다.

- `separator`: 분리 기준.
- `chunk_size`: 각 조각의 최대 길이.
- `chunk_overlap`: 조각 간 중복 길이.
- `length_function`: 텍스트 길이 측정 방식.
- `is_separator_regex`: 정규식 분리 여부.

PDF 데이터의 특수한 요구 사항에 따라 설정값을 조정하거나, 추가적인 전처리 과정을 도입하는 것이 필요할 수 있습니다.

text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)
---
### **4.2. RecursiveCharacterTextSplitter**

`RecursiveCharacterTextSplitter`는 `CharacterTextSplitter`보다 **더 유연한 텍스트 분할**을 지원하는 도구입니다.

---

 **주요 특징**
1. **`separators`**를 사용하여 분할 기준을 여러 개 설정할 수 있습니다.
   - `separators`는 리스트 형태로 제공되며, 리스트에 있는 분할 기준은 **우선순위**에 따라 적용됩니다.
   - 예를 들어:
   
     ```python
     separators = ["\n\n", "\n", " "]
     ```
     위의 설정은 먼저 `"\n\n"`을 기준으로 분할을 시도하고, 실패하면 `"\n"`을 기준으로, 마지막으로 `" "`(공백)을 기준으로 분할합니다.

2. 나머지 매개변수(`chunk_size`, `chunk_overlap`, `length_function`, `is_separator_regex`)는 `CharacterTextSplitter`와 동일하게 동작합니다.

---

 **`RecursiveCharacterTextSplitter`와 `CharacterTextSplitter`의 차이점**
| **특징**                        | **CharacterTextSplitter**                       | **RecursiveCharacterTextSplitter**              |
|---------------------------------|-------------------------------------------------|------------------------------------------------|
| **분할 기준**                  | 단일 기준 (`separator`) 사용                   | 다중 기준 (`separators` 리스트) 사용            |
| **우선순위 분할**              | 없음                                           | 리스트의 순서대로 분할 시도                     |
| **유연성**                     | 고정된 분할 기준                              | 여러 기준을 순차적으로 시도하며 더 유연         |

recursive_text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "] ,
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
### **4.3. 텍스트 분할 및 레퍼런스 제외**

전처리된 텍스트(`cleaned_doc`)를 `text_splitter`를 사용해 분할하여 `text_splits` 객체에 저장한다.  
pdf 의 경우 11페이지부터는 레퍼런스가 포함되어 있으므로, **10페이지까지만** 데이터를 불러온다.

pdf 의 경우에는 10페이지부터 있기 때문에, **9페이지까지만** 데이터를 불러온다.



pdf_text_splits = text_splitter.split_documents(cleaned_pdf)
pdf_text_splits = pdf_text_splits[:11]

pdf1_text_splits = text_splitter.split_documents(cleaned_pdf1)
pdf1_text_splits = pdf1_text_splits[1:10]


---
### **Recursive 테스트**

`RecursiveCharacterTextSplitter`를 사용해 테스트를 진행합니다.  
`merge_lines` 함수를 적용하지 않는 이유는, **recursive 방식은 한 줄씩 스플릿되기 때문에** 계단식으로 데이터가 증가하는 현상이 발생하기 때문입니다.  
또한, `RecursiveCharacterTextSplitter`는 추가적인 조정이 많이 필요해 보이며, 테스트 이후에는 `text_splits` 방식으로 데이터 처리를 이어갈 계획입니다.

text_recursive = recursive_text_splitter.split_documents(cleaned_pdf)

# 빈 리스트를 생성
filter_recursive = []

for doc in text_recursive:
    # 11페이지 까지니까 10과 같거나 작아야한다.
    if doc.metadata.get('page') <= 10:
        
        # 나오는 doc을 새로운 리스트에 할당.  
        filter_recursive.append(doc)  
        # 여기서 filter_recursive += doc하게되면 객체에 따로따로 들어가기 떄문에 변경.

---
### **4.4. merge_lines 함수**

`merge_lines` 함수는 PDF의 특성상 문장이 페이지를 넘어가면서 끊기는 문제를 해결하기 위해 만들어졌습니다.  
텍스트 분할 결과(`text_splits`)를 조정하여, **이전 청크의 마지막 줄을 현재 청크의 앞부분에 연결**합니다.

---

 **필요성**
- PDF에서 문장이 페이지를 넘어갈 때 끊기는 경우가 발생.
- `chunk_overlap`이 정상적으로 동작하지 않아 추가적인 처리가 필요.
- 이전 청크의 마지막 줄을 현재 청크의 앞부분에 연결하여 문장의 단절 문제를 해결.

---

 **구현 순서**
1. **텍스트 갯수 확인 및 반복 설정**
   - `text_splits`의 길이만큼 반복문 실행.

2. **첫 번째 청크 건너뛰기**
   - 첫 번째 텍스트(`i == 0`)는 이전 줄이 없으므로 건너뜀.

3. **이전 청크의 내용 가져오기**
   - 이전 청크(`i-1`)의 내용을 `prev_content`에 할당.

4. **이전 청크의 마지막 줄 추출**
   - `prev_content.split("\n")[-1]`로 이전 청크의 마지막 줄을 `last_line`에 저장.

5. **현재 청크의 내용 가져오기**
   - 현재 텍스트 데이터를 `text_splits[i].page_content`에서 가져옴.

6. **텍스트 연결**
   - 이전 텍스트 데이터를 현재 텍스트 데이터 앞에 붙임.

7. **결과 업데이트**
   - 연결된 텍스트로 현재 청크의 내용을 덮어씀.

def merge_lines(text):
    
    # 첫 번째 청크는 제외하기
    for i in range(1, len(text)):
        
        # 이전 청크의 마지막 줄 추출
        previous_split = text[i - 1].page_content
        
        # 마지막줄 추출하기
        last_line = previous_split.strip().splitlines()[-1]

        # 현재 청크의 텍스트 가져오기
        current_split = text[i].page_content

        # 이전 청크의 마지막 줄과 현재 청크의 첫 번째 줄을 연결
        merged_text = last_line + " " + current_split.strip()

        # 업데이트된 내용을 현재 청크에 저장
        text[i].page_content = merged_text

    return text
---
### **4.5. 텍스트 병합 및 데이터 확인**

`text_splits` 객체(스플릿된 텍스트 데이터)를 `merge_lines` 함수를 통해 처리하여 페이지 간 끊어진 문장을 연결한 새 객체 `merge_splits`에 저장합니다.  
이를 표시하여 데이터가 정제된 결과를 확인합니다.


# merge lines를 통과한 pdf데이터를 생성.
pdf_merge_splits = merge_lines(pdf_text_splits)
print("pdf file content : ")
# 각 청크마다 숫자를 매기고 \n 으로 다음줄로 그 content를 소환.
for idx, split in enumerate(pdf_merge_splits[:1]):
    print(f"스플릿 {idx}:\n{split}")

# 중간 띄워쓰기    
print("\n",{"-" * 50},"\n")

# pdf1 파일로도 생성
pdf1_merge_splits = merge_lines(pdf1_text_splits)
print("pdf1 file contetn : ")
for idx, split in enumerate(pdf1_merge_splits[:1]):
    print(f"스플릿 {idx}:\n{split}")
    

print("\n",{'-' * 50},"\n")
    
# recursive 파일로도 생성
merge_recursive = merge_lines(filter_recursive)
print("recursive file contetn : ")
for idx, split in enumerate(filter_recursive[:1]):
    print(f"스플릿 {idx}:\n{split}")


# # lambda x 는 score를 대변하며, 1에 가까울수록 정렬후, reverse = True로 변경 한 것이다.
# sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)

---
## **5. LLM 모델 적용을 위한 임베딩 생성**

LLM 모델에 데이터를 적용하기 위해 텍스트 데이터를 **임베딩(Embedding)** 해야 합니다.  
임베딩 객체를 생성하고, 필요한 모델을 설정합니다.


# OpenAI 임베딩 모델로 벡터 임베딩을 생성
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")
---
## **6. 벡터 스토어 생성**

임베딩 객체를 생성하고, 데이터를 벡터화합니다.

# FAISS 인덱스 생성
# L2(유클리드 거리) 기반으로 텍스트 벡터를 저장하고 검색하기 위한 인덱스
index = faiss.IndexFlatL2(len(embeddings.embed_query("LLM에 대해서")))

# FAISS 벡터스토어 생성
vectorstore = FAISS(
    # 임베딩 함수 지정: 텍스트를 벡터로 변환하는 함수
    embedding_function=embeddings,
    
    # 인덱스 객체: 벡터의 저장 및 검색에 사용
    index=index,
    
    # 문서 저장소: 메모리에 문서를 저장하여 인덱스와 연결
    docstore=InMemoryDocstore(),
    
    # 문서와 인덱스 간 매핑: 각 문서의 ID와 벡터 인덱스 간 관계를 저장하는 딕셔너리
    index_to_docstore_id={}
)

---
### 6.1. 백테 데이터베이스에 문서 추가

uuid를 불러와 unique한 id를 생성하고 벡터 데이터베이스에 pdf파일을 추가한다.
# pdf 문서 데이터 추가
# pdf_merge_splits의 갯수만큼, uuid 를 생성한다.
uuids = [str(uuid4()) for _ in range(len(pdf_merge_splits))]

# 벡터 데이터베이스에 각각의 문서에 id를 할당한다.
vectorstore.add_documents(documents=pdf_merge_splits, ids = uuids)


# pdf1 문서 데이터 추가
uuids = [str(uuid4()) for _ in range(len(pdf1_merge_splits))]
vectorstore.add_documents(documents=pdf1_merge_splits, ids = uuids)
---
### **6.2. 유사성 검사를 통한 문서 확인**

문서가 둘다 제대로 들어가 있는지를 알기 위해
유사성 검사를 통하여 제대로 content를 사용하는 지 확인.



벡터 데이터베이스에서 텍스트와 가장 유사한 문서를 검색하고, 점수와 함께 검색 결과를 확인한다.


# 기본 유사성 검색
# results = vectorstore.similarity_search("LLM은 어떨가요?", k=2, filter={"source": "data/LLM_Research_Trends.pdf"})
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")

#############################################################

# 점수와 함께 유사성 검색 (벡터 데이터베이스의 similarity_search_with_score을 사용)
# k 가장 유사한 문서의 갯수를 뜻함. 5면 가장 유사한 데이터 5개를 뽑음.
results_with_scores = vectorstore.similarity_search_with_score("LLM에 대해 이야기해주세요.", k=5)

# res는 검색된 문서의 객체, score는 거리이다. (점수가 낮아야 정확도 올라감)
for res, score in results_with_scores:
    
    # page_content에서 첫 10단어만 가져오기
    # " " 은 스페이스의 허용/ ""는 스페이스 없이
    limited_content = " ".join(res.page_content.split()[:10])  # 첫 10단어만 추출
    print(f"* [SIM={score:.3f}] [{limited_content} ...] {res.metadata}")

---
## **7. FAISS를 Retriever로 변환**

retriever생성한다.
벡터 데이터베이스의 as_Retriever을 사용한다.

검색의 방식을 "유사도" 로 설정하고,  
검색 결과에서 가장 유사한 문서 5개를 반환하도록 설정한다.
  
이는 5개의 유사한 문서를 토대로 답변을 생성 할 것이다. (3~5 디폴트)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
---
## **8. 프롬프트 템플릿의 정의**

컨텍스트 데이터를 기반으로 질문에 답변을 생성하기 위한 **프롬프트 템플릿**이다.  
이 프롬프트는 다음과 같은 특징을 가지고 있다.

---
 **프롬프트의 구성**
1. **시스템 메시지**:
   - 답변은 제공된 컨텍스트 데이터만 사용하도록 제한.
   - 질문의 입력 언어와 상관없이 **항상 한국어**로 답변 생성.
   - 컨텍스트 데이터에 없는 질문일 경우, 관련된 질문을 다시 요청하도록 안내.

2. **사용자 메시지**:
   - 질문과 컨텍스트 데이터를 함께 입력.
   - 컨텍스트와 질문을 분리하여 명확하게 전달.

---

 **사용 목적**
- 모델이 제공된 **컨텍스트 데이터** 내에서만 질문에 대한 답변을 생성하도록 유도.
- 답변의 언어를 한국어로 고정하여 일관성을 유지.
- 질문이 컨텍스트 데이터와 관련 없는 경우, 답변 대신 **다른 질문을 요청**함으로써 모델의 한계를 명확히 전달.


이 프롬프트는 컨텍스트 데이터 기반의 응답 생성에 최적화되어 있으며, 제한된 범위 내에서 신뢰성 있는 답변을 보장합니다.

# contextual_prompt = ChatPromptTemplate.from_messages([
#     # 시스템 메시지: 답변 규칙 정의
#     ("system", "Answer the following question using only given context data."),
#     ("system", "Answer must be given in Korean, regardless of the input language."),
#     ("system", "If the question is outside the given context data, suggest asking something related."),

    
#     ("system", "Here is an example of how to answer a question:\n"
#                "User: LLM에 대해서 알려줘\n"
#                "AI: LLM은 대규모 언어 모델로, 자연어 처리 작업에 활용됩니다.\n"
#                "Follow this format when generating responses."),
    
    
#     ("system", "답변은 ~다. 로 끊지않고 친근한 말투를 사용해줘.\n"
#                "User: LLM에 대해서 알려줘\n"
#                "AI: LLM은 대규모 언어 모델로, 자연어 처리 작업에 활용되고 있어.\n"
#                "Follow this format when generating responses"),
    
#     # 한 문장이 끝날 때 줄바꿈 요청
#     ("system", "답변은 반드시 한 문장이 끝날 때마다 줄바꿈('\n')을 추가해서 작성해.\n"
#                "이 규칙을 모든 문장에 적용해.\n"
#                "예를 들어:\n"
#                "User: LLM에 대해서 알려줘\n"
#                "AI: LLM은 대규모 언어 모델을 의미해.\n"
#                "주로 자연어 처리 작업에 사용되며,\n"
#                "번역, 요약, 질의응답 등 다양한 작업을 수행할 수 있어.\n"
#                "프롬프트를 어떻게 입력하느냐에 따라,\n"
#                "다양한 능력을 발휘할 수 있지.\n"
#                "Follow this format when generating responses"),
    
#     # 사용자 메시지 템플릿
#     ("user", "Context: {context}\\n\\nQuestion: {question}")
# ])

---
# 도전 과제를 위한 불러오기
# 파일 경로와 시스템 메시지 파일 리스트
path = "Prompts/"
system_files = ["prompt1.txt", "prompt2.txt", "prompt3.txt", "prompt4.txt"]

# 시스템 메시지 불러오기
system_messages = []
for txt in system_files:
    with open(os.path.join(path, txt), "r", encoding="UTF-8") as f:
        
         # \\n을 \n으로 변환
        content = f.read().replace("\\n", "\n") 
        system_messages.append(("system", content))


# 사용자 메시지 템플릿 추가
system_messages.append(("user", "Context: {context}\\n\\nQuestion: {question}"))

# ChatPromptTemplate 생성
contextual_prompt = ChatPromptTemplate.from_messages(system_messages)
contextual_prompt

---
## **9. RAG 체인 구성**


이 코드는 **RAG(Retrieval-Augmented Generation) 체인**의 디버깅과 문서 데이터를 모델에 적합한 형태로 변환하기 위해 설계되었습니다.  
각 단계에서 데이터가 올바르게 전달되고 처리되는지 확인할 수 있습니다.

---

 **1. `DebugPassThrough` 클래스**
- 데이터를 전달하면서 디버깅을 수행하는 클래스.
- **주요 역할**:
  1. 입력 데이터를 그대로 전달.
  2. 데이터가 정상적으로 전달되는지 확인 가능.
- **특징**:
  - `*args, **kwargs`를 사용해 유연하게 입력값을 처리.
  - 테스트 시 출력(`print`)을 활성화하여 데이터 확인 가능.

---

 **2. `ContextToText` 클래스**
- **문서 리스트를 하나의 텍스트로 변환**하여 모델 입력에 적합한 형태로 가공.
- **주요 역할**:
  1. 입력된 문서 리스트(`inputs["context"]`)의 각 문서(`page_content`)를 줄바꿈(`\n`)으로 결합.
  2. 컨텍스트(`context`)와 질문(`question`)을 분리하여 반환.

  #### **2. ContextToText 클래스**

`ContextToText` 클래스는 문서 리스트를 하나의 텍스트로 변환하여 모델 입력에 적합한 형태로 가공합니다.

- **주요 기능**:
  - 입력된 문서 리스트의 `page_content`를 줄바꿈(`\n`)으로 결합.
  - 결과를 `context`와 `question`으로 나누어 반환.

---

 **3. DebugPassThrough 클래스**

`DebugPassThrough` 클래스는 데이터를 디버깅하기 위해 설계된 도구입니다.  
입력 데이터를 그대로 전달하면서, 필요 시 데이터를 확인할 수 있습니다.

- **주요 기능**:
  - 데이터를 전달하는 과정에서 디버깅 용도로 출력 가능.
  - 데이터의 흐름을 점검하고, 전달 과정에서 발생할 수 있는 문제를 식별.

- **특징**:
  - `*args`와 `**kwargs`를 사용하여 유연하게 데이터를 처리.
  - 테스트 시 출력(`print`)을 활성화하여 디버깅.

---

 **4. RAG 체인 구성**

RAG 체인은 검색, 디버깅, 데이터 변환, 프롬프트 생성, 모델 호출 단계를 포함하여 데이터의 흐름을 체계적으로 처리합니다.



# 데이터가 정상적으로 전달되는지 확인을 위한 디버깅 클래스
class DebugPassThrough(RunnablePassthrough):
    
    # 어떠한 값이 올지 모르니 , *args, **kwargs를 사용하여 유기적으로 받을수 있도록 설정.
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        
        # 프린트가 안이뻐 보여서 테스트 제외하고는 주석처리
        # 받아오는 값을 그대로 출력한다.
        # print("Debug Output:", output)
        return output
    
# 문서 리스트를 텍스트로 변환하는 단계 추가 (모델에 적합한 형태로 가공)
class ContextToText(RunnablePassthrough):
    
    def invoke(self, inputs, config=None, **kwargs):
        
        # 불러온 각 문서의 page_content를 출력하고 줄바꿈으로 결합하여 하나의 텍스트로 변경.
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        
        # 출력할때, content와 question을 분리해서 반환한다.
        return {"context": context_text, "question": inputs["question"]}
    

# RAG 체인에서 각 단계마다 DebugPassThrough 추가
rag_chain_debug = {
    
    # 컨텍스트 문서에서 검색을 하는 retriever
    "context": retriever,      
    
    # 유저의 인풋이 제대로 받는지 디버깅 과정을 통함              
    "question": DebugPassThrough() 
    
    # RAG체인의 순서를 설정. Retriever > Debug > context > prompt > model순으로 데이터 전달.      
}  | DebugPassThrough() | ContextToText()|   contextual_prompt | model
---
## **10. 챗봇 구동**


이 코드는 **RAG(Retrieval-Augmented Generation)** 체인을 사용하여 질문에 답변을 생성하는 루프 기반 질의응답 시스템입니다.  
사용자가 질문을 입력하면, 관련 문서를 검색하고 답변을 생성한 뒤, 검색된 문서의 유사도 점수를 계산합니다.

---

 **1. 동작 흐름**

1. **사용자 입력**:
   - 사용자로부터 질문을 입력받습니다.
   - **"break"** 입력 시 루프를 종료합니다.

2. **RAG 체인 호출**:
   - 사용자 입력(`query`)을 RAG 체인(`rag_chain_debug`)에 전달하여 답변을 생성합니다.

3. **유사성 검색 및 점수 계산**:
   - 입력된 질문과 가장 관련 있는 상위 5개의 문서를 검색(`similarity_search_with_score`).
   - 검색된 문서들의 유사도 점수를 추출하고, 평균 점수를 계산합니다.

4. **결과 출력**:
   - 입력된 질문, 유사도 평균 점수, 그리고 최종 답변을 출력합니다.

---

 **주요 기능 설명**

1. **사용자 입력**:
   - **`input`**을 통해 질문을 입력받습니다.
   - **"break"**를 입력하면 루프를 종료하여 프로그램을 종료할 수 있습니다.

2. **RAG 체인 호출**:
   - **`rag_chain_debug.invoke(query)`**:
     - 입력된 질문을 RAG 체인에 전달하여 검색된 문서를 기반으로 답변을 생성합니다.
     - RAG 체인은 검색(`retriever`)과 데이터 변환(`ContextToText`) 및 LLM 호출을 포함한 파이프라인입니다.

3. **유사성 검색 및 점수 계산**:
   - **`vectorstore.similarity_search_with_score(query, k=5)`**:
     - 질문과 가장 관련성이 높은 상위 5개의 문서를 검색.
     - 검색 결과는 문서와 유사도 점수(`score`)로 반환됩니다.
   - **유사도 점수 추출 및 평균 계산**:
     - 검색된 문서의 유사도 점수만 추출하여 평균값을 계산(**`mean(scores)`**).
     - 평균 점수는 검색된 문서와 질문 간의 관련성을 평가하는 척도입니다.

4. **결과 출력**:
   - **`print`**를 사용해 다음 정보를 출력합니다:
     1. **`query`**: 사용자 입력 질문.
     2. **`average_score`**: 검색된 문서와 질문 간 유사도 평균 점수.
     3. **`response.content`**: RAG 체인이 생성한 최종 답변.

---

 **요약**

- **RAG 체인 기반 질의응답 시스템**:
  - 사용자의 질문에 대해 관련 문서를 검색하고, 문서를 기반으로 LLM을 활용해 답변을 생성합니다.

- **유사도 점수**:
  - 검색된 문서와 질문 간의 관련성을 평가하기 위해 유사도 점수를 계산합니다.
  - 평균 점수를 통해 검색된 결과의 신뢰성을 확인할 수 있습니다.

- **간단한 루프 구조**:
  - 프로그램이 사용자의 입력을 지속적으로 처리하며, **"break"** 입력 시 종료됩니다.
  
이 시스템은 사용자와의 상호작용을 통해 검색 및 생성 모델의 능력을 효과적으로 활용합니다.


# while True: 
#     print("========================")
    
#     # 사용자의 입력을 받음
#     query = input("질문을 입력하세요 (break 입력시 종료): ")

#     # "break" 입력 시 루프 종료
#     if query.lower() == "break":
#         break

#     # 위에서 설정한 RAG chaing을 invoke즉 불로오고 값으로 유저의 인풋 query를 매개변수로 보냄.
#     response = rag_chain_debug.invoke(query)
    
#     # 점수와 함께 유사성 검색 (상위 5개 문서)
#     results_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    
#     # 유사성 점수만 추출
#     scores = [score for _, score in results_with_scores]

#     # 평균 계산
#     average_score = mean(scores)
    
#     print("Question : ", query)
#     print("Distance : ", average_score)
#     print("Final Response:")
#     print(response.content)
---
# 도전 과제를 위한 프린트 path로 저장하기

# 결과 저장 경로 설정
output_path = "Results/"

# 결과 저장 경로가 없으면 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)

while True: 
    print("========================")
    
    # 사용자의 입력을 받음
    query = input("질문을 입력하세요 (break 입력시 종료): ")

    # "break" 입력 시 루프 종료
    if query.lower() == "break":
        break

    # 위에서 설정한 RAG chain을 invoke 즉 불러오고 값으로 유저의 인풋 query를 매개변수로 보냄.
    response = rag_chain_debug.invoke(query)
    
    # 점수와 함께 유사성 검색 (상위 5개 문서)
    results_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    
    # 유사성 점수만 추출
    scores = [score for _, score in results_with_scores]

    # 평균 계산
    average_score = mean(scores)
    
    # 출력 결과
    print("Question : ", query)
    print("Distance : ", average_score)
    print("Final Response:")
    print(response.content)
    
    
    ###################################### 동일
    

    # 파일 이름에 현재 시간 추가
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_name = f"result_{timestamp}.txt"
    file_path = os.path.join(output_path, file_name)

    # 결과를 파일로 저장
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Question: {query}\n")
        f.write(f"Distance: {average_score:.3f}\n")
        f.write("Final Response:\n")
        f.write(response.content)
    
    print(f"Result saved to: {file_path}")

