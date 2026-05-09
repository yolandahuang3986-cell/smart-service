"""
投诉Agent v2 — 集成情绪·意图双层护栏
业务逻辑：情绪三级感知 + 安抚话术 + 人工升级
护栏逻辑：由注入的 GuardRail 实例承担，通过私有方法调用
"""
from typing import Dict, List, Optional
from .base_agent import BaseAgent, Message, AgentResponse, IntentType


# 情绪安抚话术库
# 结构：情绪类型 → 话术列表
# 合并了组员版本的场景细分逻辑，在原有四类情绪基础上扩充
SOOTHE_TEMPLATES = {
    # ── 情绪类型模板（opener 选择依据）────────────────────
    "anger": [
        "非常抱歉给您带来了不好的体验，我完全理解您现在的心情。",
        "您的感受完全可以理解，我们有责任为您解决这个问题。",
        "这确实是我们服务的失误，您的愤怒完全合理，请给我一次机会处理。",
    ],
    "threat": [
        "您的诉求我已经认真记录，会第一时间为您跟进处理。",
        "我理解您希望问题尽快得到解决，请允许我为您安排优先处理。",
        "您有权通过任何方式维权，同时我们也希望能在这里为您解决问题。",
    ],
    "agitation": [
        "非常抱歉让您有这样的感受，请告诉我具体发生了什么。",
        "您的反馈对我们很重要，我来帮您认真处理这个问题。",
        "我能感受到您的失望，请相信我会认真对待您的每一个诉求。",
    ],
    "general": [
        "感谢您联系我们，我会尽力帮助您解决问题。",
        "我理解您的情况，让我来帮您处理。",
        "非常抱歉给您带来不便，请告诉我具体情况，我来跟进。",
    ],
}

# ── 场景细分话术（body 部分，按问题类型匹配）────────────────
# 关键词命中后追加在 opener 之后，提供具体解决方向
# 来源：合并组员版本 solution_templates，保持与护栏逻辑解耦
SCENARIO_TEMPLATES = {
    "delivery_delay": {
        "triggers": ["没到", "慢", "等太久", "延迟", "还没收到", "几天了", "什么时候到"],
        "response": (
            "\n\n关于配送延迟，我会立即协助您核查物流状态。"
            "如确认延误超出承诺时效，可申请配送补偿或优先重新发货。"
            "请提供您的订单号，我来为您查询。"
        ),
    },
    "quality_issue": {
        "triggers": ["质量", "坏", "破", "瑕疵", "故障", "坏掉", "损坏", "有问题"],
        "response": (
            "\n\n质量问题我们承担全责，您可以选择：\n"
            "① 全额退款  ② 换货补发  ③ 部分补偿\n"
            "请告知您的偏好，我来为您发起对应申请。"
        ),
    },
    "service_issue": {
        "triggers": ["态度", "客服", "不理", "不回复", "没人管", "已读不回", "爱答不理"],
        "response": (
            "\n\n对于服务态度问题我深表歉意，会将您的反馈记录并反馈给服务主管，"
            "并亲自跟进您的问题直到解决。这不是我们应有的服务水准。"
        ),
    },
    "wrong_item": {
        "triggers": ["发错", "不对", "少发", "漏发", "错发", "不是我要的"],
        "response": (
            "\n\n发错货是我们的失误，我们会立即安排补发正确商品。"
            "错误商品您可以选择退回（运费我们承担）或保留，请告知您的偏好。"
        ),
    },
    "refund_delay": {
        "triggers": ["退款没到", "退款慢", "退款多久", "还没退", "等了很久退款"],
        "response": (
            "\n\n退款到账通常需要 3-7 个工作日，具体取决于支付方式。"
            "请提供订单号，我来为您核查退款进度，如有异常立即处理。"
        ),
    },
    "no_response": {
        "triggers": ["没有回音", "没人理", "联系不上", "打不通", "一直没消息"],
        "response": (
            "\n\n非常抱歉让您长时间等待无回应，这是我们的服务失误。"
            "我现在为您优先处理，请告知您等待回复的具体问题，我来直接跟进。"
        ),
    },
}


class ComplaintAgent(BaseAgent):

    def __init__(self, guardrail=None):
        """
        Args:
            guardrail: GuardRail 实例，由 Orchestrator 注入。
                       为 None 时护栏静默跳过，保持向后兼容。
        """
        super().__init__("complaint", "ComplaintAgent")
        self._guardrail = guardrail
        # 会话级投诉计数：session_id → count
        self._complaint_counts: Dict[str, int] = {}

    # ══════════════════════════════════════════
    # 主处理入口
    # ══════════════════════════════════════════

    def process(self, message: Message) -> AgentResponse:
        content      = message.content
        session_id   = message.session_id
        data         = message.data
        emotion_info = data.get("emotion_level", {})
        emotion_score: int = emotion_info.get("score", 0)
        emotion_level: str = emotion_info.get("level", "low")

        self.log(f"处理投诉: emotion={emotion_level}({emotion_score}) | {content[:40]}")

        # ── 获取会话历史（用于渐进式越狱检测）──────────
        session_history: List[Dict] = data.get("session_history", [])

        # ══ Layer 1 · 输入层护栏 ══════════════════════
        guard_result = self._run_input_guard(
            content, emotion_score, session_history
        )
        if guard_result and guard_result.blocked:
            return AgentResponse(
                success=False,
                message=guard_result.block_reason,
                need_escalate=guard_result.escalate_to_human,
                escalate_reason="护栏拦截：" + "、".join(guard_result.triggered_rules),
                data={
                    "action": "blocked",
                    "guard": guard_result.to_dict(),
                },
            )

        # ══ Layer 2 · 处理层 — 敏感操作拦截 ══════════
        sensitive_block = self._run_sensitive_guard(content, emotion_score)
        if sensitive_block:
            return AgentResponse(
                success=False,
                message=sensitive_block,
                data={"action": "sensitive_blocked"},
            )

        # ── 投诉计数（升级判断依据之一）────────────────
        count = self._increment_complaint_count(session_id)

        # ── 情绪三级分流 ─────────────────────────────
        if self._should_escalate(emotion_score, emotion_level, content, count):
            return self._escalate(emotion_score, content, guard_result)

        # ── 生成安抚响应 ──────────────────────────────
        response_text = self._generate_soothe(
            content, emotion_level, emotion_score, count
        )

        # ══ Layer 3 · 输出层合规复审 ══════════════════
        response_text = self._run_output_guard(response_text, emotion_score)

        return AgentResponse(
            success=True,
            message=response_text,
            data={
                "action": "soothe",
                "emotion_level": emotion_level,
                "emotion_score": emotion_score,
                "complaint_count": count,
                "guard": guard_result.to_dict() if guard_result else {},
            },
        )

    # ══════════════════════════════════════════
    # 护栏调用（私有方法，隔离护栏与业务）
    # ══════════════════════════════════════════

    def _run_input_guard(self, content, emotion_score, history):
        if self._guardrail is None:
            return None
        try:
            result = self._guardrail.check_input(content, emotion_score, history)
            if result.triggered_rules:
                self.log(
                    f"[Guard·输入层] level={result.risk_level} "
                    f"score={result.rule_score:.1f} "
                    f"rules={result.triggered_rules} "
                    f"sdk={result.sdk_invoked}"
                )
            return result
        except Exception as e:
            self.log(f"[Guard·输入层] 异常（{e}），跳过")
            return None

    def _run_sensitive_guard(self, content, emotion_score) -> Optional[str]:
        if self._guardrail is None:
            return None
        try:
            return self._guardrail.check_sensitive_operation(content, emotion_score)
        except Exception as e:
            self.log(f"[Guard·处理层] 异常（{e}），跳过")
            return None

    def _run_output_guard(self, response: str, emotion_score: int) -> str:
        if self._guardrail is None:
            return response
        try:
            return self._guardrail.check_output(response, emotion_score)
        except Exception as e:
            self.log(f"[Guard·输出层] 异常（{e}），跳过")
            return response

    # ══════════════════════════════════════════
    # 情绪三级感知业务逻辑
    # ══════════════════════════════════════════

    def _should_escalate(
        self,
        emotion_score: int,
        emotion_level: str,
        content: str,
        complaint_count: int,
    ) -> bool:
        """
        升级人工触发条件（任一满足）：
          1. 情绪分 ≥ 8
          2. 含威胁性词汇（曝光/投诉/12315/法院）
          3. 明确要求"转人工/找经理"
          4. 同会话重复投诉 ≥ 3 次
        """
        if emotion_score >= 8:
            return True
        threat_words = ["曝光", "投诉", "12315", "法院", "律师", "起诉", "媒体"]
        if any(w in content for w in threat_words):
            return True
        escalate_words = ["转人工", "找经理", "要人工", "人工客服", "不要机器人"]
        if any(w in content for w in escalate_words):
            return True
        if complaint_count >= 3:
            return True
        return False

    def _escalate(
        self, emotion_score: int, content: str, guard_result
    ) -> AgentResponse:
        """触发人工升级。"""
        guard_info = guard_result.to_dict() if guard_result else {}
        progressive = guard_info.get("progressive_jailbreak", False)

        reason_parts = []
        if emotion_score >= 8:
            reason_parts.append(f"情绪分={emotion_score}")
        if progressive:
            reason_parts.append("渐进式越狱检测命中")
        if any(w in content for w in ["曝光", "投诉", "12315"]):
            reason_parts.append("威胁性词汇")
        if any(w in content for w in ["转人工", "找经理"]):
            reason_parts.append("明确要求人工")

        escalate_reason = "、".join(reason_parts) if reason_parts else "高风险会话"

        msg = (
            "非常抱歉给您带来了不好的体验！\n\n"
            "您的问题已被列为优先处理，正在为您安排专属人工客服，"
            "预计 5 秒内接入，请稍候。\n\n"
            "感谢您的耐心，我们一定妥善解决您的问题。"
        )
        return AgentResponse(
            success=True,
            message=msg,
            need_escalate=True,
            escalate_reason=escalate_reason,
            data={
                "action": "escalate",
                "emotion_score": emotion_score,
                "guard": guard_info,
            },
        )

    def _generate_soothe(
        self,
        content: str,
        emotion_level: str,
        emotion_score: int,
        complaint_count: int,
    ) -> str:
        """
        根据情绪等级生成分级安抚话术。
        结构：opener（情绪类型匹配）+ scenario body（场景细分）+ level body（情绪等级引导）
        """
        import random

        # ── 1. opener：按情绪类型选模板 ──────────────────
        if any(kw in content for kw in ["曝光", "投诉", "起诉", "律师", "12315", "法院"]):
            template_key = "threat"
        elif any(kw in content for kw in ["气死", "愤怒", "骗子", "欺诈", "混蛋", "垃圾"]):
            template_key = "anger"
        elif emotion_level == "medium":
            template_key = "agitation"
        else:
            template_key = "general"

        opener = random.choice(SOOTHE_TEMPLATES[template_key])

        # ── 2. scenario body：场景细分，匹配具体问题 ─────
        scenario_body = ""
        for scenario, cfg in SCENARIO_TEMPLATES.items():
            if any(t in content for t in cfg["triggers"]):
                scenario_body = cfg["response"]
                break  # 只取第一个命中的场景，避免叠加过长

        # ── 3. level body：情绪等级引导语 ────────────────
        if scenario_body:
            # 已有场景细分，level body 只做情绪承接，不重复引导
            if emotion_level == "high":
                level_body = "\n\n我已将您的问题列为优先处理，请稍候。"
            elif emotion_level == "medium":
                level_body = "\n\n我会认真跟进，争取给您一个满意的答复。"
            else:
                level_body = ""
        else:
            # 无场景细分时，level body 承担引导用户描述问题的职责
            if emotion_level == "high":
                level_body = (
                    "\n\n我注意到您情绪比较激动，我非常理解。"
                    "请您告诉我具体是哪个订单出现了问题，"
                    "我来为您优先核实并给出解决方案。"
                )
            elif emotion_level == "medium":
                level_body = (
                    "\n\n请告诉我您遇到的具体问题，"
                    "我会认真帮您处理，争取给您一个满意的答复。"
                )
            else:
                level_body = "\n\n请告诉我您的具体情况，我来帮您解决。"

        # ── 4. 多次投诉加强安抚 ──────────────────────────
        repeat_note = ""
        if complaint_count >= 2:
            repeat_note = (
                "\n\n我注意到您已经联系我们多次，"
                "非常抱歉还没有解决您的问题，我会重点跟进。"
            )

        return opener + scenario_body + level_body + repeat_note

    # ══════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════

    def _increment_complaint_count(self, session_id: str) -> int:
        self._complaint_counts[session_id] = (
            self._complaint_counts.get(session_id, 0) + 1
        )
        return self._complaint_counts[session_id]
