"""
智能客服系统 - 中央协调器
v4: 新增 GuardRail 注入 + session_history 透传
"""
import time
import uuid
from typing import Dict, Optional, List
from datetime import datetime

from agents import (
    RouterAgent, OrderAgent, LogisticsAgent,
    RefundAgent, ComplaintAgent,
    Message, AgentResponse, IntentType,
)
from knowledge import KnowledgeRetriever
from evaluation import SessionJudge
from guardrail import GuardRail

# Router 置信度低于此值视为 fallback
ROUTER_CONFIDENCE_THRESHOLD = 0.6


class CustomerServiceOrchestrator:
    """
    中央协调器 v4

    新增功能（在 v3 基础上）：
      - GuardRail 统一创建，注入 ComplaintAgent 和 RefundAgent
      - process_message() 在分发前将 session_history 挂载到 message.data，
        供 ComplaintAgent 渐进式越狱检测使用
      - stats 新增 guard_blocks（护栏拦截次数）
    """

    def __init__(self, build_index_on_start: bool = False):
        # ── 检索器 ──────────────────────────────
        self.retriever = KnowledgeRetriever(score_threshold=0.60)
        if build_index_on_start:
            print("🔍 预热向量索引...")
            self.retriever.build_index()
            print("✅ 向量索引构建完成")

        # ── 护栏（共享实例）─────────────────────
        self.guardrail = GuardRail()

        # ── Agent 初始化 ─────────────────────────
        self.router          = RouterAgent()
        self.order_agent     = OrderAgent()
        self.logistics_agent = LogisticsAgent(retriever=self.retriever)
        self.refund_agent    = RefundAgent(retriever=self.retriever)
        self.complaint_agent = ComplaintAgent(guardrail=self.guardrail)

        self.agents = {
            "router":    self.router,
            "order":     self.order_agent,
            "logistics": self.logistics_agent,
            "refund":    self.refund_agent,
            "complaint": self.complaint_agent,
        }

        # ── Judge ────────────────────────────────
        self.judge       = SessionJudge()
        self.evaluations: Dict[str, Dict] = {}

        self.sessions: Dict[str, Dict] = {}
        self.stats = {
            "total_requests":     0,
            "escalation_count":   0,
            "fallback_count":     0,
            "guard_blocks":       0,      # 护栏拦截次数（新增）
            "intent_distribution": {i.value: 0 for i in IntentType},
            "rag_hits":    0,
            "rag_sources": {},
        }

    # ── 主处理流程 ────────────────────────────────────────

    def process_message(self, user_input: str, session_id: str = None) -> Dict:
        if not session_id or session_id not in self.sessions:
            session_id = self._create_session()
        session = self.sessions[session_id]
        session["messages"].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        self.stats["total_requests"] += 1

        print(f"\n{'='*60}")
        print(f"📝 用户输入: {user_input}")
        print(f"🆔 会话ID: {session_id}")
        print(f"{'='*60}\n")

        # ── Step 1: Router（含计时）────────────────────────
        router_msg = Message(
            sender="user", receiver="router",
            intent=IntentType.UNKNOWN,
            content=user_input,
            data={
                # 透传历史窗口 + 上轮情绪分给 RouterAgent
                "session_history":   session["messages"],
                "prev_emotion_score": session["context"]["last_emotion_score"],
            },
            session_id=session_id,
        )
        t0 = time.perf_counter()
        router_resp = self.router.receive_message(router_msg)
        router_latency = (time.perf_counter() - t0) * 1000

        intent     = router_resp.data.get("intent", "unknown")
        confidence = router_resp.data.get("confidence", 0.0)
        emotion    = router_resp.data.get("emotion_level", {})

        # ── 更新 session context：实体缓存（旧值保留，新值补充）──
        ctx = session["context"]
        new_entities = router_resp.data.get("extracted_data", {})
        for field in ("order_id", "tracking_number", "phone", "refund_reason"):
            if new_entities.get(field):          # 有新值才覆盖
                ctx[field] = new_entities[field]

        # ── 更新情绪加权分（供下一轮 Router 使用）────────────
        ctx["last_emotion_score"] = emotion.get("score", 0.0)

        # fallback 判断：Router 置信度低于阈值
        router_fallback = confidence < ROUTER_CONFIDENCE_THRESHOLD

        print(
            f"🤖 [RouterAgent] 意图={intent}  置信度={confidence:.2f}"
            f"{'  ⚠️ fallback' if router_fallback else ''}  "
            f"情绪={emotion.get('level', 'low')}({emotion.get('score', 0)})  "
            f"耗时={router_latency:.0f}ms  → {router_resp.next_agent}\n"
        )

        if intent in self.stats["intent_distribution"]:
            self.stats["intent_distribution"][intent] += 1

        # ── Step 2: 业务 Agent（含计时）───────────────────
        target_id = router_resp.next_agent
        target    = self.agents.get(target_id)
        if not target:
            return self._error(session_id, "找不到对应处理 Agent")

        biz_msg = Message(
            sender="router", receiver=target_id,
            intent=IntentType(intent),
            content=user_input,
            data={
                **router_resp.data,
                # ── context 合并：将缓存实体注入 extracted_data ──
                # 规则：context 里有值但本轮未提取到时，用 context 补充
                "extracted_data": {
                    **ctx,                                       # context 缓存作为基底
                    **router_resp.data.get("extracted_data", {}), # 本轮新提取的值优先
                },
                "session_context":  ctx,        # 完整 context 供 Agent 按需使用
                "session_history":  session["messages"],  # 历史消息供渐进式检测使用
            },
            session_id=session_id,
        )
        t1 = time.perf_counter()
        biz_resp = target.receive_message(biz_msg)
        biz_latency = (time.perf_counter() - t1) * 1000

        # fallback 判断：RAG 降级到关键词匹配
        rag_fallback = biz_resp.rag_used and biz_resp.data.get("rag_fallback", False)
        fallback_triggered = router_fallback or rag_fallback

        # 写入 Trace 层计算的 latency（Agent 自身不计时）
        biz_resp.latency_ms          = biz_latency
        biz_resp.fallback_triggered  = fallback_triggered

        if fallback_triggered:
            self.stats["fallback_count"] += 1

        rag_flag = "✅" if biz_resp.rag_used else "—"
        print(
            f"🤖 [{target.name}] 成功={biz_resp.success}  "
            f"升级={biz_resp.need_escalate}  "
            f"RAG={rag_flag}  fallback={fallback_triggered}  "
            f"耗时={biz_latency:.0f}ms\n"
        )

        # ── Trace 埋点：写入 session ────────────────────────
        trace = {
            "turn":                  len(session["traces"]) + 1,
            "timestamp":             datetime.now().isoformat(),
            "user_input":            user_input,
            "intent":                intent,
            "router_confidence":     confidence,
            "router_latency_ms":     round(router_latency, 1),
            "router_fallback":       router_fallback,
            "router_emotion_score":  emotion.get("score", 0),
            "router_emotion_raw":    emotion.get("raw_score", 0),   # 新增：本轮原始分
            "router_emotion_level":  emotion.get("level", "low"),
            "context_snapshot":      {k: v for k, v in ctx.items() if k != "last_emotion_score"},
            "target_agent":          target_id,
            "biz_latency_ms":        round(biz_latency, 1),
            "biz_success":           biz_resp.success,
            "need_escalate":         biz_resp.need_escalate,
            "rag_used":              biz_resp.rag_used,
            "rag_sources":           biz_resp.rag_sources,
            "rag_fallback":          rag_fallback,
            "fallback_triggered":    fallback_triggered,
        }
        session["traces"].append(trace)

        # ── 护栏拦截统计 ────────────────────────────────────
        if biz_resp.data.get("action") == "blocked":
            self.stats["guard_blocks"] += 1
        if biz_resp.rag_used:
            self.stats["rag_hits"] += 1
            for src in biz_resp.rag_sources:
                self.stats["rag_sources"][src] = self.stats["rag_sources"].get(src, 0) + 1

        if biz_resp.need_escalate:
            self.stats["escalation_count"] += 1
            session["status"] = "escalated"

        session["messages"].append({
            "role":        "assistant",
            "content":     biz_resp.message,
            "agent":       target_id,
            "success":     biz_resp.success,
            "need_escalate": biz_resp.need_escalate,
            "rag_used":    biz_resp.rag_used,
            "rag_sources": biz_resp.rag_sources,
            "timestamp":   datetime.now().isoformat(),
        })

        return {
            "success":          True,
            "session_id":       session_id,
            "response":         biz_resp.message,
            "intent":           intent,
            "agent":            target_id,
            "need_escalate":    biz_resp.need_escalate,
            "escalate_reason":  biz_resp.escalate_reason,
            "data":             biz_resp.data,
            "rag_used":         biz_resp.rag_used,
            "rag_sources":      biz_resp.rag_sources,
            "latency_ms":       round(biz_latency, 1),
            "fallback_triggered": fallback_triggered,
        }

    # ── 会话评估（手动触发）────────────────────────────────

    def evaluate_session(self, session_id: str) -> Dict:
        """
        对指定会话运行 LLM-as-Judge 评估，结果写入 self.evaluations。

        使用方式：
            result = orchestrator.evaluate_session("abc12345")
            print(result["quadrant"], result["action_desc"])
        """
        session = self.sessions.get(session_id)
        if not session:
            return {"error": f"session {session_id} 不存在"}

        result = self.judge.evaluate(session)
        self.evaluations[session_id] = result

        print(
            f"\n📊 [Judge] {session_id} → {result['quadrant']}\n"
            f"   resolved={result['resolved']}  satisfied={result['satisfied']}\n"
            f"   情绪趋势={result['emotion_trend']}\n"
            f"   动作={result['action_desc']}\n"
            f"   理由={result['judge_reasoning']}\n"
        )
        return result

    # ── 会话管理 ──────────────────────────────

    def _create_session(self) -> str:
        sid = str(uuid.uuid4())[:8]
        self.sessions[sid] = {
            "session_id": sid,
            "created_at": datetime.now().isoformat(),
            "status":     "active",
            "messages":   [],
            "traces":     [],
            "context":    {
                # ── 实体缓存：旧值保留，新值补充 ──────────
                "order_id":        None,
                "tracking_number": None,
                "phone":           None,
                "refund_reason":   None,
                # ── 情绪滑动加权状态 ────────────────────
                "last_emotion_score": 0.0,   # 上轮加权后的情绪分，初始值 0
            },
        }
        print(f"✨ 创建新会话: {sid}")
        return sid

    def _error(self, session_id: str, msg: str) -> Dict:
        return {
            "success": False,
            "session_id": session_id,
            "response": f"抱歉，系统出现错误：{msg}",
            "intent": "error",
            "agent": None,
            "need_escalate": True,
            "escalate_reason": "系统错误",
            "data": {},
            "rag_used": False,
            "rag_sources": [],
        }

    # ── 统计 ──────────────────────────────────

    def get_stats(self) -> Dict:
        total      = self.stats["total_requests"]
        rag_rate   = (self.stats["rag_hits"] / total * 100) if total else 0
        fb_rate    = (self.stats["fallback_count"] / total * 100) if total else 0
        eval_count = len(self.evaluations)
        resolved_count   = sum(1 for e in self.evaluations.values() if e.get("resolved"))
        satisfied_count  = sum(1 for e in self.evaluations.values() if e.get("satisfied"))
        return {
            **self.stats,
            "total_sessions":     len(self.sessions),
            "active_sessions":    sum(1 for s in self.sessions.values() if s["status"] == "active"),
            "escalated_sessions": sum(1 for s in self.sessions.values() if s["status"] == "escalated"),
            "rag_hit_rate":       f"{rag_rate:.1f}%",
            "fallback_rate":      f"{fb_rate:.1f}%",
            "guard_block_rate":   f"{self.stats['guard_blocks']/total*100:.1f}%" if total else "N/A",
            "retriever_stats":    self.retriever.get_stats(),
            # ── 评估汇总 ──
            "evaluated_sessions": eval_count,
            "resolve_rate":       f"{resolved_count/eval_count*100:.1f}%" if eval_count else "N/A",
            "satisfaction_rate":  f"{satisfied_count/eval_count*100:.1f}%" if eval_count else "N/A",
            "quadrant_distribution": self._quadrant_distribution(),
        }

    def _quadrant_distribution(self) -> Dict:
        dist = {}
        for e in self.evaluations.values():
            q = e.get("quadrant", "unknown")
            dist[q] = dist.get(q, 0) + 1
        return dist


# ── 单例 ───────────────────────────────────────

_orchestrator: Optional[CustomerServiceOrchestrator] = None

def get_orchestrator(build_index_on_start: bool = False) -> CustomerServiceOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CustomerServiceOrchestrator(
            build_index_on_start=build_index_on_start
        )
    return _orchestrator
