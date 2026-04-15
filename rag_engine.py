import os
# 强制将 HuggingFace 设为国内镜像，解决下载模型卡死的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import yaml
import chromadb
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
    import jieba  # 如果文档可能有中文，使用 jieba 分词效果更好
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except (ImportError, OSError) as e:
    print(f"[RAG] 警告: 加载 sentence_transformers 失败 ({e})。系统将自动降级，不使用 Reranker 重排序。")
    CrossEncoder = None

class RAGEngine:
    def __init__(self, config_path: str = "config/kotaemon.yaml"):
        """
        初始化包含混合检索 (向量 + BM25) 及重排序模块的 RAG 引擎
        模拟了 Kotaemon 的核心 Indexer 和 Retriever 工作流
        """
        self._load_config(config_path)
        self._init_vector_store()
        self._init_keyword_store()
        self._init_reranker()
        
    def _load_config(self, config_path: str):
        # 尝试基于工作目录加载配置文件
        # 兼容独立运行或全局运行
        if not os.path.exists(config_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "rjxmgl", "config", "kotaemon.yaml")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.chunk_size = self.config.get("chunking", {}).get("chunk_size", 512)
        self.top_k_initial = self.config.get("retriever", {}).get("top_k_initial", 10)
        self.top_k_final = self.config.get("reranker", {}).get("top_k_final", 5)

    def _init_vector_store(self):
        v_store = self.config.get("vector_store", {})
        persist_dir = v_store.get("persist_directory", "./data/chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        # 禁用遥测防止卡住
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False)
        )

        print("[RAG] 正在连接/创建向量集合 (初次可能从 HF 镜像下载模型，请稍等一阵子)...")
        # 恢复使用 ChromaDB 原生的嵌入模型 (all-MiniLM-L6-v2)
        self.collection = self.client.get_or_create_collection(
            name=v_store.get("collection_name", "papers_collection"),
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[RAG] 向量引擎初始化成功: 现有 {self.collection.count()} 个文档块")

    def _init_keyword_store(self):
        # 初始化基于内存的 BM25 模型
        self.bm25_model = None
        self.bm25_corpus = []          # 存放原本分词后的结构
        self.bm25_documents = []       # 存放原文及元数据
        
        if BM25Okapi is None:
            print("[RAG] 警告: 未安装 rank_bm25。将降级为仅向量检索。 (pip install rank_bm25 jieba)")
        else:
            self._rebuild_bm25_index()

    def _init_reranker(self):
        rerank_cfg = self.config.get("reranker", {})
        self.reranker = None
        enable_rerank = rerank_cfg.get("enabled", False)
        
        if enable_rerank:
            if CrossEncoder is not None:
                model_name = rerank_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                print(f"[RAG] 正在加载 Reranker 模型 {model_name} ...")
                self.reranker = CrossEncoder(model_name)
            else:
                print("[RAG] 警告: 未安装 sentence_transformers。Reranker 被禁用。 (pip install sentence-transformers)")
    
    def _tokenize(self, text: str) -> List[str]:
        """针对中文/英文的简单混合分词"""
        if "jieba" in globals():
            return list(jieba.cut(text))
        return text.lower().split()

    def _rebuild_bm25_index(self):
        """内部方法：重新建立或加载 BM25 本地全量索引"""
        if BM25Okapi is None: return
        
        # 简单模拟: 从 chroma db 拉取全量
        try:
            all_docs = self.collection.get()
            if not all_docs["documents"]:
                print("[RAG] BM25索引为空。")
                return
        except Exception:
            return
            
        self.bm25_documents = []
        tokenized_corpus = []
        for i, text in enumerate(all_docs["documents"]):
            tokenized_corpus.append(self._tokenize(text))
            self.bm25_documents.append({
                "content": text,
                "metadata": all_docs["metadatas"][i] if all_docs["metadatas"] else {},
                "id": all_docs["ids"][i]
            })
            
        self.bm25_model = BM25Okapi(tokenized_corpus)
        print(f"[RAG] BM25全文检索缓存建立完毕。")

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        统一入库接口，接收成员 D 处理出来的 chunks
        chunk 格式: {"content": "...", "metadata": {"paper_title": "...", "year": 2023}, "vector": [...](可选)}
        """
        if not chunks:
            return False

        texts, metadatas, embeddings, ids = [], [], [], []
        base_idx = self.collection.count()
        
        for i, chunk in enumerate(chunks):
            texts.append(chunk.get("content", ""))
            
            # 清理元数据中的 None，转为合法类型
            safe_meta = {k: v for k, v in chunk.get("metadata", {}).items() if v is not None}
            metadatas.append(safe_meta)
            
            if "vector" in chunk and chunk["vector"] is not None:
                embeddings.append(chunk["vector"])
                
            chunk_id = chunk.get("id", f"doc_chunk_{base_idx + i}")
            ids.append(chunk_id)

        try:
            # 存入向量库
            if len(embeddings) == len(texts):
                self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
            else:
                self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
                
            print(f"[RAG] 成功入库 {len(texts)} 个块。")
            
            # 刷新或者增量追加 BM25
            self._rebuild_bm25_index()
            return True
        except Exception as e:
            print(f"[RAG] 入库失败: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        暴露给系统的统一检索接口: (Hybrid Search + Rerank)
        """
        initial_k = self.top_k_initial
        candidates = []
        
        # 1. 向量召回 (Vector Search)
        vector_results = []
        try:
            v_res = self.collection.query(query_texts=[query], n_results=initial_k)
            if v_res and v_res["documents"] and len(v_res["documents"][0]) > 0:
                for idx in range(len(v_res["documents"][0])):
                    vector_results.append({
                        "content": v_res["documents"][0][idx],
                        "metadata": v_res["metadatas"][0][idx] if v_res["metadatas"] else {},
                        "id": v_res["ids"][0][idx],
                        "v_score": 1.0 - (v_res["distances"][0][idx] if v_res["distances"] else 0.0)
                    })
        except Exception as e:
            print(f"[RAG] 向量检索报错: {e}")

        # 2. 文本召回 (BM25)
        bm25_results = []
        if self.bm25_model is not None:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:initial_k]
            for idx in top_n:
                if bm25_scores[idx] > 0:
                    doc = self.bm25_documents[idx].copy()
                    doc["b_score"] = bm25_scores[idx]
                    bm25_results.append(doc)

        # 3. 混合去重与合并
        candidate_map = {}
        for r in vector_results:
            candidate_map[r["id"]] = r
            
        for r in bm25_results:
            if r["id"] in candidate_map:
                candidate_map[r["id"]]["b_score"] = r.get("b_score", 0.0)
            else:
                candidate_map[r["id"]] = r

        candidates = list(candidate_map.values())
        if not candidates:
            return []

        # 4. 重排序 (Reranker) 或 RRF 平替计算 score
        if self.reranker is not None:
            pairs = [[query, doc["content"]] for doc in candidates]
            scores = self.reranker.predict(pairs)
            for i, doc in enumerate(candidates):
                doc["score"] = float(scores[i])
        else:
            for doc in candidates:
                v = doc.get("v_score", 0.0)
                b = doc.get("b_score", 0.0)
                doc["score"] = 0.6 * v + 0.4 * min(b / 10.0, 1.0)
                
        # 按分数排序并截断
        candidates.sort(key=lambda x: x["score"], reverse=True)
        final_results = candidates[:min(top_k, self.top_k_final)]

        # 返回约定的标准化 JSON 列表
        return [
            {
                "content": res["content"],
                "metadata": res.get("metadata", {}),
                "score": round(res["score"], 4)
            } for res in final_results
        ]

