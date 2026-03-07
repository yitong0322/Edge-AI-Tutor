#!/usr/bin/env python3
"""
ingest_pdf.py — 用 fastembed 建立向量数据库
模型: BAAI/bge-small-en-v1.5 (本地 ONNX，不走 Ollama)
"""

import os
import time
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding

# ============================================================
# 配置（路径和集合名与 main.py 完全一致）
# ============================================================
PROJECT_DIR     = "/home/user/AITutor"
DATA_PATH       = os.path.join(PROJECT_DIR, "data")
DB_PATH         = os.path.join(PROJECT_DIR, "chroma_db")
COLLECTION_NAME = "electronics_knowledge"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE      = 300   # 缩小片段，减少 RAG prompt 的 prefill tokens
CHUNK_OVERLAP   = 30


# ============================================================
# FastEmbed → LangChain Embeddings 适配器
# ============================================================
class FastEmbedEmbeddings(Embeddings):
    """把 fastembed.TextEmbedding 包装成 LangChain Embeddings 接口"""

    def __init__(self, model_name: str = FASTEMBED_MODEL):
        print(f"[Embed] Loading fastembed model: {model_name}")
        self.model = TextEmbedding(model_name)
        print("[Embed] Model ready (ONNX, runs in-process)")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0].tolist()


# ============================================================
# 主流程
# ============================================================
def create_vector_db():
    if not os.path.exists(PROJECT_DIR):
        print(f"❌ 找不到主文件夹: {PROJECT_DIR}")
        return

    # 1. 检查数据文件夹
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"⚠️  {DATA_PATH} 不存在，已创建。请放入 PDF 后再运行。")
        return

    print(f"--- [1/4] 扫描 PDF: {DATA_PATH} ---")
    try:
        loader    = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()
    except ImportError:
        print("❌ 缺少 pypdf: pip install pypdf --break-system-packages")
        return

    if not documents:
        print("❌ 没有找到 PDF 文件！")
        return
    print(f"✅ 加载了 {len(documents)} 页")

    # 2. 切分
    print("--- [2/4] 切分文本 ---")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ 切分为 {len(chunks)} 个片段")

    # 3. 建立向量数据库
    print(f"--- [3/4] 生成向量并存入: {DB_PATH} ---")
    embeddings = FastEmbedEmbeddings()
    t0 = time.time()

    vector_db = Chroma.from_documents(
        documents       = chunks,
        embedding       = embeddings,
        persist_directory= DB_PATH,
        collection_name = COLLECTION_NAME,
    )

    print(f"✅ 数据库构建完成！耗时: {time.time()-t0:.1f}s")
    print(f"   集合: {COLLECTION_NAME}")
    print(f"   路径: {DB_PATH}")

    # 4. 验证
    print("--- [4/4] 验证检索 ---")
    results = vector_db.similarity_search("operational amplifier", k=1)
    if results:
        print(f"✅ 验证通过，首条结果: {results[0].page_content[:80]}...")
    else:
        print("⚠️  检索无结果，请检查 PDF 内容")


if __name__ == "__main__":
    create_vector_db()