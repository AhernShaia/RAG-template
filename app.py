import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
# 內建模組
import mimetypes
import os
# 專案模組
from langchain_openai import ChatOpenAI, AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Qdrant

# loader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredXMLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 環境變量
from dotenv import load_dotenv
from pydantic import BaseModel

# 定義 Pydantic 模型來解析請求的 JSON 資料


class URLRequest(BaseModel):
    url: str


load_dotenv()

# azure embedding info
azure_embeddings_api_key = os.getenv('AZURE_EMBEDDINGS_API_KEY')
azure_embeddings_deployment = os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_EMBEDDINGS_ENDPOINT')
azure_openai_version = os.getenv('AZURE_OPENAI_VERSION')


def vector_conversion(data, loader):
    # 調用embedding模型向量化
    embedding_model = AzureOpenAIEmbeddings(
        api_key=azure_embeddings_api_key,
        azure_deployment=azure_embeddings_deployment,
        openai_api_version=azure_openai_version,
        azure_endpoint=azure_openai_endpoint,
    )
    # chunk
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    data = loader.load()
    chunks = spliter.split_documents(data)

    # 2. 將向量儲存
    qdrant = Qdrant.from_documents(
        chunks,
        embedding_model,
        collection_name="local_documents",
        url=os.getenv('QDRANT_CLIENT_URL'),
        api_key=os.getenv('QDRANT_CLIENT_API_KEY'),
        timeout=60,
    )
    if qdrant:
        print("向量化完成")
        return {"status": 'success', "message": "向量化完成"}
    else:
        print("向量化失敗")
        return {"status": 'failed', "message": "向量化失敗"}


# ----------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------------------------

# 設定文件上傳的目錄
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 設定靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")
# 根路徑返回index.html


@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = Path("static/index.html")
    if index_path.exists():
        return index_path.read_text()
    else:
        return {"message": "index.html not found"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 檢查文件類型
        if file.content_type not in ["application/pdf", "text/csv", "text/xml"]:
            raise HTTPException(
                status_code=400, detail="Unsupported file type")

        # 設定文件保存路徑
        file_location = UPLOAD_DIR / file.filename

        # 保存文件到服務器
        with file_location.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 獲取文件的 MIME 類型
        mime_type, _ = mimetypes.guess_type(file_location)
        print('*'*50)
        print(mime_type)
        print('*'*50)
        # 處理 CSV 文件
        if mime_type == "application/pdf":
            print('PDF file uploaded successfully')
            loader = PyPDFLoader(file_path=str(
                file_location), extract_images=True)
            print('loading data...')
            data = loader.load()
            print('splitting data')
            return vector_conversion(data=data, loader=loader)

        elif mime_type == 'text/csv':
            print('CSV file uploaded successfully')
            # 1. loader
            loader = CSVLoader(file_path=str(file_location))
            print('loading data...')
            data = loader.load()
            print('splitting data')
            return vector_conversion(data=data, loader=loader)

        else:
            # print(mime_type)
            # print(file_location)
            print('XML file uploaded successfully')
            # 1. loader
            print('loading data...')
            loader = UnstructuredXMLLoader(file_path=str(file_location))
            data = loader.load()
            # print(data[0].page_content)
            print('splitting data')
            return vector_conversion(data=data, loader=loader)

    # 處理文件上傳時的異常
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"File upload failed: {str(e)}, only PDF, CSV, XML files are supported.")


@app.post("/add_url")
async def add_url(request: URLRequest):
    url = request.url
    # 1. loader
    loader = WebBaseLoader(url)
    data = loader.load()
    return vector_conversion(data=data, loader=loader)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
