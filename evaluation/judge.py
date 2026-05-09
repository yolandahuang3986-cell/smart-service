"""
LLM-as-Judge 会话质量评估器
手动调用：orchestrator.evaluate_session(session_id)

评估维度：
  - resolved:   是否解决（二元，Judge 判断）
  - satisfied:  是否满意（Judge 推断 + 末轮情绪分辅助）
  - emotion_trend: 情绪趋势定性（好转 / 平稳 / 恶化）
  - quadrant:   四象限标签 + recommended_action
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 四象限动作表 ──────────────────────────────────────────
QUADRANT_ACTIONS: Dict[str, Dict] = {
    "resolved_satisfied": {
        "label": "✅ 已解决·满意",
        "action": "archive",
        "action_desc": "存档",
    },
    "resolved_unsatisfied": {
        "label": "⚠️ 已解决·不满意",
        "action": "review_solution",
        "action_desc": "推送解决方案人工岗穿测 + 知识库复核",
    },
    "unresolved_satisfied": {
        "label": "⚠️ 未解决·满意",
        "action": "review_kb",
        "action_desc": "内存标记 + 知识库规则复核",
    },
    "unresolved_unsatisfied": {
        "label": "❌ 未解决·不满意",
        "action": "human_followup",
        "action_desc": "内存标记 + 推送人工回访",
    },
}

# ── 情绪趋势阈值 ──────────────────────────────────────────
EMOTION_IMPROVE_THRESHOLD = -2   # 分数下降 ≥ 2 → 好转
EMOTION_WORSEN_THRESHOLD  =  2   # 分数上升 ≥ 2 → 恶化


class SessionJudge:
    """
    会话质量评估器

    使用示例：
        judge = SessionJudge()
        result = judge.evaluate(session)
    """

    JUDGE_MODEL = "claude-sonnet-4-20250514"  # 复用 Anthropic API（与架构一致）

    def __init__(self):
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── 主入口 ────────────────────────────────────────────

    def evaluate(self, session: Dict) -> Dict:
        """
        对一个完整会话做质量评估。

        Args:
            session: orchestrator.sessions[sid] 的完整字典

        Returns:
            evaluation dict，包含所有维度结果和 recommended_action
        """
        messages   = session.get("messages", [])
        traces     = session.get("traces", [])
        emotion_scores = self._extract_emotion_scores(traces)

        # ── 1. 情绪趋势（纯计算，不调用 LLM）──────────────
        emotion_trend = self._calc_emotion_trend(emotion_scores)

        # ── 2. LLM 判断 resolved + satisfied ──────────────
        resolved, satisfied, judge_reasoning = self._llm_judge(
            messages, emotion_scores
        )

        # ── 3. 四象限归类 ──────────────────────────────────
        quadrant_key = f"{'resolved' if resolved else 'unresolved'}_{'satisfied' if satisfied else 'unsatisfied'}"
        quadrant     = QUADRANT_ACTIONS[quadrant_key]

        result = {
            "session_id":       session["session_id"],
            "resolved":         resolved,
            "satisfied":        satisfied,
            "emotion_trend":    emotion_trend,
            "emotion_scores":   emotion_scores,
            "quadrant":         quadrant["label"],
            "recommended_action": quadrant["action"],
            "action_desc":      quadrant["action_desc"],
            "judge_reasoning":  judge_reasoning,
            "evaluated_at":     _now_iso(),
        }

        logger.info(
            f"[Judge] session={session['session_id']} "
            f"resolved={resolved} satisfied={satisfied} "
            f"quadrant={quadrant['label']} action={quadrant['action']}"
        )
        return result

    # ── 情绪趋势计算 ──────────────────────────────────────

    def _extract_emotion_scores(self, traces: List[Dict]) -> List[int]:
        """从 trace 列表里提取各轮的情绪分数。"""
        scores = []
        for t in traces:
            score = t.get("router_emotion_score")
            if score is not None:
                scores.append(score)
        return scores

    def _calc_emotion_trend(self, scores: List[int]) -> str:
        if len(scores) < 2:
            return "平稳"
        delta = scores[-1] - scores[0]
        if delta <= EMOTION_IMPROVE_THRESHOLD:
            return "好转"
        if delta >= EMOTION_WORSEN_THRESHOLD:
            return "恶化"
        return "平稳"

    # ── LLM-as-Judge ─────────────────────────────────────

    def _llm_judge(
        self,
        messages: List[Dict],
        emotion_scores: List[int],
    ) -> tuple[bool, bool, str]:
        """
        调用 LLM 判断 resolved 和 satisfied。
        返回 (resolved, satisfied, reasoning)
        """
        conversation_text = self._format_conversation(messages)
        last_emotion = emotion_scores[-1] if emotion_scores else None

        prompt = f"""你是一个客服质量评估专家。请根据以下对话判断两个问题，并以 JSON 格式回答。

对话记录：
{conversation_text}

末轮情绪分数（0-10，越高越负面）：{last_emotion if last_emotion is not None else '未知'}

请判断：
1. resolved（是否解决）：用户的核心问题是否得到了实质性解答或处理？
   - 判断依据：用户明确确认（如"好的谢谢"/"确认退款"），或末轮未触发升级人工
   - true = 已解决，false = 未解决

2. satisfied（是否满意）：用户在对话结束时情绪是否平和/正面？
   - 判断依据：对话内容中的情绪表达 + 末轮情绪分（< 3 视为满意辅助信号）
   - 两者均低才判定 false（不满意）
   - true = 满意，false = 不满意

只返回 JSON，不要任何其他文字：
{{"resolved": true或false, "satisfied": true或false, "reasoning": "简要说明判断依据，不超过50字"}}"""

        try:
            result = self._call_llm(prompt)
            data = json.loads(result)
            return (
                bool(data.get("resolved", False)),
                bool(data.get("satisfied", True)),
                data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"[Judge] LLM 调用失败（{e}），使用规则降级判断")
            return self._rule_fallback(messages, emotion_scores)

    def _call_llm(self, prompt: str) -> str:
        """调用 Anthropic API。"""
        import urllib.request

        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
            return body["content"][0]["text"]

    # ── 规则降级（LLM 不可用时）──────────────────────────

    def _rule_fallback(
        self, messages: List[Dict], emotion_scores: List[int]
    ) -> tuple[bool, bool, str]:
        """
        LLM 不可用时的保底判断：
        - resolved：末轮 assistant 消息 success=True 且未触发升级
        - satisfied：末轮情绪分 < 3
        """
        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"), {}
        )
        resolved  = (last_assistant.get("success", False)
                     and not last_assistant.get("need_escalate", False))
        last_score = emotion_scores[-1] if emotion_scores else 5
        satisfied  = last_score < 3
        return resolved, satisfied, "规则降级判断（LLM 不可用）"

    # ── 工具 ─────────────────────────────────────────────

    def _format_conversation(self, messages: List[Dict]) -> str:
        lines = []
        for m in messages:
            role = "用户" if m.get("role") == "user" else "客服"
            lines.append(f"{role}：{m.get('content', '')}")
        return "\n".join(lines)


# ── 模块级工具函数 ────────────────────────────────────────

def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
