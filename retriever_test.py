import json
from rag_engine import RAGEngine

def run_tests():
    print("=== 初始化 RAG 引擎 ===")
    engine = RAGEngine()
    
    # 模拟成员 D 处理好的入库文档结构
    mock_chunks = [
        {
            "content": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with a text generator model. It can help reduce hallucinations.",
            "metadata": {"paper_title": "RAG Foundations", "authors": "Lewis et al.", "year": 2020},
            "id": "doc_rag_1"
        },
        {
            "content": " BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document, regardless of their proximity.",
            "metadata": {"paper_title": "Information Retrieval Classics", "authors": "Robertson", "year": 2009},
            "id": "doc_bm25_1"
        },
        {
            "content": "In this paper we propose a hybrid search mechanism that combines BM25 and dense vector embeddings (like ChromaDB) followed by a cross-encoder reranker for superior accuracy.",
            "metadata": {"paper_title": "Hybrid Search in Modern RAG", "authors": "Smith", "year": 2023},
            "id": "doc_hybrid_1"
        },
        {
            "content": "Quantum computing utilizes the principles of quantum mechanics.",
            "metadata": {"paper_title": "Quantum Limits", "authors": "Feynman", "year": 1982},
            "id": "doc_qc_1"
        }
    ]

    print("\n=== 测试入库接口 (add_documents) ===")
    success = engine.add_documents(mock_chunks)
    assert success is True, "文档入库失败"

    # 测试检索
    queries = [
        "How can we reduce hallucinations in text generation?",
        "Tell me about bag-of-words and BM25",
        "Hybrid search with cross-encoder and vector embeddings"
    ]

    print("\n=== 测试统一检索接口 (retrieve) ===")
    for query in queries:
        print(f"\n[Query]: {query}")
        results = engine.retrieve(query, top_k=2)
        
        for i, res in enumerate(results):
            print(f"  [{i+1}] Score: {res['score']:.4f} | Title: {res['metadata'].get('paper_title')}")
            print(f"      Text:  {res['content']}")

if __name__ == "__main__":
    run_tests()
