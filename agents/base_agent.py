"""
智能客服多Agent系统 - 基础Agent类
v2: 注入 KnowledgeRetriever，支持 Agentic RAG
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import re
import time


class IntentType(Enum):
    """意图类型枚举"""
    ORDER = "order"
    LOGISTICS = "logistics"
    REFUND = "refund"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


@dataclass
class Message:
    """Agent间通信消息"""
    sender: str
    receiver: str
    intent: IntentType
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "intent": self.intent.value,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }


@dataclass
class AgentResponse:
    """Agent响应结果"""
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    need_escalate: bool = False
    escalate_reason: str = ""
    next_agent: Optional[str] = None
    # ── RAG 字段 ──────────────────────────────
    rag_used: bool = False
    rag_sources: List[str] = field(default_factory=list)
    # ── 可观测性新增字段 ───────────────────────
    latency_ms: float = 0.0          # 由 Orchestrator Trace 层写入，Agent 无需自填
    fallback_triggered: bool = False  # Router 低置信度路由 OR RAG 降级关键词匹配

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "need_escalate": self.need_escalate,
            "escalate_reason": self.escalate_reason,
            "next_agent": self.next_agent,
            "rag_used": self.rag_used,
            "rag_sources": self.rag_sources,
            "latency_ms": self.latency_ms,
            "fallback_triggered": self.fallback_triggered,
        }


class BaseAgent(ABC):
    """
    Agent 基类 v2

    新增 RAG 支持：
      - __init__ 接收可选的 retriever 参数
      - 提供 retrieve_knowledge() 便捷方法，子类按需调用
      - 所有现有子类无需修改即可兼容（retriever 默认为 None）
    """

    def __init__(self, agent_id: str, name: str, retriever=None):
        """
        Args:
            agent_id: Agent 唯一标识
            name: Agent 显示名称
            retriever: KnowledgeRetriever 实例（可选）。
                       由 Orchestrator 在初始化时统一注入，Agent 内部只使用，不负责创建。
        """
        self.agent_id = agent_id
        self.name = name
        self.message_history: List[Message] = []
        self._retriever = retriever  # 由外部注入，保持 Agent 轻量

    # ── 核心抽象方法 ──────────────────────────

    @abstractmethod
    def process(self, message: Message) -> AgentResponse:
        pass

    # ── 消息收发 ──────────────────────────────

    def receive_message(self, message: Message) -> AgentResponse:
        self.message_history.append(message)
        return self.process(message)

    def send_message(self, receiver: str, intent: IntentType,
                     content: str, data: Dict[str, Any] = None,
                     session_id: str = "") -> Message:
        return Message(
            sender=self.agent_id,
            receiver=receiver,
            intent=intent,
            content=content,
            data=data or {},
            session_id=session_id,
        )

    # ── RAG 便捷方法 ──────────────────────────

    def retrieve_knowledge(
        self,
        query: str,
        top_k: int = 2,
        category_filter: str = None,
    ):
        """
        检索知识库，返回 List[RetrievalResult]。
        若未注入 retriever，静默返回空列表，不影响现有逻辑。

        Args:
            query: 检索问题（通常为用户原始输入或关键意图）
            top_k: 返回条数
            category_filter: 可选，限定类别（"refund" / "logistics"）

        Returns:
            List[RetrievalResult]，未注入 retriever 时返回 []
        """
        if self._retriever is None:
            return []
        try:
            results = self._retriever.retrieve(
                query, top_k=top_k, category_filter=category_filter
            )
            if results:
                self.log(
                    f"[RAG] 检索到 {len(results)} 条知识: "
                    + ", ".join(f"{r.doc_id}({r.score:.2f})" for r in results)
                )
            return results
        except Exception as e:
            self.log(f"[RAG] 检索异常（{e}），跳过 RAG")
            return []

    def format_rag_context(self, results, max_chars: int = 600) -> str:
        """
        将检索结果格式化为上下文字符串。
        若未注入 retriever 或结果为空，返回空字符串。
        """
        if not results or self._retriever is None:
            return ""
        return self._retriever.format_context(results, max_chars=max_chars)

    # ── 工具方法 ──────────────────────────────

    def log(self, message: str):
        print(f"[{self.name}] {message}")
