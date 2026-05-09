"""
知识库检索器 — Agentic RAG 核心组件

使用 Google text-embedding-004 将 FAQ 文档向量化，
通过余弦相似度检索最相关的知识片段，零外部向量数据库依赖。

依赖：
    pip install google-generativeai numpy
"""

from __future__ import annotations

import os
import math
import time
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 懒加载，避免在没有 API Key 的环境中直接报错 ──
_genai = None

def _get_genai():
    global _genai
    if _genai is None:
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "未设置 GOOGLE_API_KEY 环境变量。"
                "请执行: export GOOGLE_API_KEY='your_key'"
            )
        genai.configure(api_key=api_key)
        _genai = genai
    return _genai


# ──────────────────────────────────────────────
# 向量工具函数（纯 Python + math，无 numpy 依赖）
# ──────────────────────────────────────────────

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度，范围 [-1, 1]，越大越相似。"""
    denom = _norm(a) * _norm(b)
    if denom == 0:
        return 0.0
    return _dot(a, b) / denom


# ──────────────────────────────────────────────
# 检索结果数据类
# ──────────────────────────────────────────────

class RetrievalResult:
    """单条检索结果"""
    def __init__(self, doc: Dict, score: float):
        self.doc_id: str = doc["id"]
        self.category: str = doc["category"]
        self.question: str = doc["question"]
        self.answer: str = doc["answer"]
        self.score: float = score          # 余弦相似度

    def __repr__(self):
        return f"<RetrievalResult id={self.doc_id} score={self.score:.3f}>"

    def to_context_str(self) -> str:
        """格式化为可直接插入 Prompt 的文本片段。"""
        return (
            f"【参考知识 · {self.question}】\n"
            f"{self.answer}\n"
            f"(相关度: {self.score:.2f})"
        )


# ──────────────────────────────────────────────
# 主检索器
# ──────────────────────────────────────────────

class KnowledgeRetriever:
    """
    Agentic RAG 检索器

    工作流程：
        1. 初始化时对所有 FAQ 文档做一次 embedding（带缓存）
        2. 查询时对 query 做 embedding，与文档库做余弦相似度排序
        3. 返回 top-k 结果，供 Agent 注入 Prompt

    使用示例：
        retriever = KnowledgeRetriever()
        results = retriever.retrieve("质量问题怎么退款", top_k=2)
        for r in results:
            print(r.to_context_str())
    """

    EMBED_MODEL = "models/text-embedding-004"
    # 查询任务类型：RETRIEVAL_QUERY 适合用户问题，RETRIEVAL_DOCUMENT 适合建索引
    TASK_QUERY    = "RETRIEVAL_QUERY"
    TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"

    def __init__(
        self,
        documents: Optional[List[Dict]] = None,
        score_threshold: float = 0.60,
        use_keyword_fallback: bool = True,
    ):
        """
        Args:
            documents: FAQ 文档列表，默认加载 faq_store.FAQ_DOCUMENTS
            score_threshold: 相似度低于此值的结果不返回（0~1）
            use_keyword_fallback: 当 embedding 失败时，降级为关键词匹配
        """
        if documents is None:
            from .faq_store import FAQ_DOCUMENTS
            documents = FAQ_DOCUMENTS

        self.documents = documents
        self.score_threshold = score_threshold
        self.use_keyword_fallback = use_keyword_fallback

        # 文档向量缓存: doc_id → List[float]
        self._doc_vectors: Dict[str, List[float]] = {}
        self._indexed = False

        logger.info(f"KnowledgeRetriever 初始化，共 {len(documents)} 条文档")

    # ── 索引构建 ──────────────────────────────

    def build_index(self) -> None:
        """
        对所有文档做 embedding 并缓存。
        首次 retrieve() 时自动调用，也可手动提前构建。
        """
        if self._indexed:
            return

        logger.info("开始构建文档向量索引...")
        texts = [f"{d['question']} {d['answer']}" for d in self.documents]

        try:
            vectors = self._batch_embed(texts, task_type=self.TASK_DOCUMENT)
            for doc, vec in zip(self.documents, vectors):
                self._doc_vectors[doc["id"]] = vec
            self._indexed = True
            logger.info(f"索引构建完成，共 {len(self._doc_vectors)} 条向量")
        except Exception as e:
            logger.warning(f"向量索引构建失败（{e}），将使用关键词降级检索")
            self._indexed = False  # 保持 False，触发关键词 fallback

    # ── 检索入口 ──────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        检索最相关的 FAQ 文档。

        Args:
            query: 用户问题或意图描述
            top_k: 返回条数
            category_filter: 可选，限定类别（"refund" / "logistics"）

        Returns:
            按相关度降序排列的 RetrievalResult 列表
        """
        if not self._indexed:
            self.build_index()

        # 过滤文档
        docs = self.documents
        if category_filter:
            docs = [d for d in docs if d["category"] == category_filter]

        # 优先走向量检索
        if self._indexed and self._doc_vectors:
            return self._vector_search(query, docs, top_k)

        # 降级：关键词匹配
        if self.use_keyword_fallback:
            logger.info("使用关键词降级检索")
            return self._keyword_search(query, docs, top_k)

        return []

    # ── 向量检索 ──────────────────────────────

    def _vector_search(
        self, query: str, docs: List[Dict], top_k: int
    ) -> List[RetrievalResult]:
        try:
            query_vec = self._embed_single(query, task_type=self.TASK_QUERY)
        except Exception as e:
            logger.warning(f"Query embedding 失败（{e}），降级到关键词检索")
            return self._keyword_search(query, docs, top_k) if self.use_keyword_fallback else []

        scored: List[Tuple[float, Dict]] = []
        for doc in docs:
            doc_vec = self._doc_vectors.get(doc["id"])
            if doc_vec is None:
                continue
            score = cosine_similarity(query_vec, doc_vec)
            if score >= self.score_threshold:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [RetrievalResult(doc, score) for score, doc in scored[:top_k]]

    # ── 关键词降级检索 ────────────────────────

    def _keyword_search(
        self, query: str, docs: List[Dict], top_k: int
    ) -> List[RetrievalResult]:
        """简单关键词命中计数，作为 embedding 不可用时的保底方案。"""
        scored: List[Tuple[float, Dict]] = []
        for doc in docs:
            hits = sum(1 for kw in doc.get("keywords", []) if kw in query)
            if hits > 0:
                # 归一化到 [0, 1] 区间，方便与向量分数统一比较
                score = min(hits / max(len(doc.get("keywords", [1])), 1), 1.0)
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [RetrievalResult(doc, score) for score, doc in scored[:top_k]]

    # ── Google Embedding API 调用 ─────────────

    def _embed_single(self, text: str, task_type: str) -> List[float]:
        genai = _get_genai()
        resp = genai.embed_content(
            model=self.EMBED_MODEL,
            content=text,
            task_type=task_type,
        )
        return resp["embedding"]

    def _batch_embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        """
        批量 embedding，每批最多 100 条（Google API 限制），
        批次间加小延迟避免速率限制。
        """
        genai = _get_genai()
        all_vectors: List[List[float]] = []
        batch_size = 20

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = genai.embed_content(
                model=self.EMBED_MODEL,
                content=batch,
                task_type=task_type,
            )
            all_vectors.extend(resp["embedding"])
            if i + batch_size < len(texts):
                time.sleep(0.2)  # 避免触发 QPS 限制

        return all_vectors

    # ── 工具方法 ──────────────────────────────

    def format_context(
        self, results: List[RetrievalResult], max_chars: int = 800
    ) -> str:
        """
        将检索结果格式化为 Prompt 上下文字符串。

        Args:
            results: retrieve() 的返回值
            max_chars: 最大字符数，超出时截断最后一条

        Returns:
            可直接拼入 system prompt 或 user message 的字符串
        """
        if not results:
            return ""

        parts = ["以下是相关知识库内容，请参考回答：\n"]
        total = len(parts[0])

        for i, r in enumerate(results, 1):
            chunk = f"\n{i}. {r.to_context_str()}\n"
            if total + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total += len(chunk)

        return "".join(parts)

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "indexed_documents": len(self._doc_vectors),
            "is_indexed": self._indexed,
            "score_threshold": self.score_threshold,
        }
