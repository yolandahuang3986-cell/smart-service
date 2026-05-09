"""
路由Agent - 判断问题类型并分发到对应Agent
"""
import re
from typing import Dict, List
from .base_agent import BaseAgent, Message, AgentResponse, IntentType


class RouterAgent(BaseAgent):
    """RouterAgent：分析用户输入，判断意图类型"""

    def __init__(self):
        super().__init__("router", "RouterAgent")
        self.intent_patterns = {
            IntentType.ORDER: {
                "keywords": [
                    "订单", "下单", "购买", "买了", "商品", "查订单", "我的订单",
                    "订单号", "订单状态", "改地址", "修改地址", "换地址", "地址错了",
                    "取消订单", "不要了",
                ],
                "patterns": [
                    r"订单[号编号]?\s*\d+",
                    r"改[变更].*?地址",
                    r"地址.*?改",
                    r"查[询看].*?订单",
                ],
            },
            IntentType.LOGISTICS: {
                "keywords": [
                    "物流", "快递", "发货", "配送", "送到", "收货", "到哪了",
                    "什么时候到", "多久到", "查物流", "查快递", "跟踪", "进度",
                    "快递员", "驿站", "派送",
                ],
                "patterns": [
                    r"[A-Z]{2}\d{9,13}",
                    r"[货包裹].*?到哪[里啦]",
                    r"物流[信息状态]",
                ],
            },
            IntentType.REFUND: {
                "keywords": [
                    "退款", "退货", "退钱", "退单", "申请退款", "我要退", "不想买了",
                    "质量问题", "坏了", "瑕疵", "不符", "七天无理由", "不喜欢", "不合适",
                ],
                "patterns": [
                    r"[申请]*退[款货]",
                    r"[能可].*?退[吗嘛]",
                    r"怎么退",
                    r"质量[问题有]",
                ],
            },
            IntentType.COMPLAINT: {
                "keywords": [
                    "投诉", "举报", "差评", "不满", "生气", "愤怒", "太差了",
                    "垃圾", "坑人", "骗子", "服务态度", "没人管", "不负责任",
                    "曝光", "媒体", "12315", "消费者协会", "气死了", "太失望了",
                    "忍无可忍", "经理", "主管", "领导", "转人工", "找人工",
                ],
                "patterns": [
                    r"太[差烂].*?了",
                    r"[气恼].*?[死炸]",
                    r"曝光.*?你们",
                    r"[经理主管领导].*?[来出见]",
                ],
            },
        }
        self.emotion_keywords = {
            "angry":       ["气", "怒", "火", "炸", "死", "滚", "垃圾", "坑", "骗", "差", "烂"],
            "urgent":      ["急", "快", "马上", "立刻", "必须", "赶紧"],
            "disappointed": ["失望", "后悔", "不该", "上当", "受骗"],
        }

    def process(self, message: Message) -> AgentResponse:
        content = message.content
        data    = message.data
        self.log(f"分析用户输入: {content}")

        # ── 历史窗口（最近 2 轮用户消息）────────────────────
        history: list = data.get("session_history", [])
        recent_user_msgs = [
            m["content"] for m in history
            if m.get("role") == "user"
        ][-2:]

        # ── 意图识别（结合历史窗口辅助判断）────────────────
        intent_scores = self._analyze_intent(content, recent_user_msgs)
        best_intent   = max(intent_scores, key=intent_scores.get)
        best_score    = intent_scores[best_intent]
        self.log(f"意图识别结果: {best_intent.value} (置信度: {best_score:.2f})")

        # ── 实体提取（当前轮）────────────────────────────────
        extracted_data = self._extract_entities(content, best_intent)

        # ── 情绪检测 + 滑动加权 ──────────────────────────────
        current_emotion = self._detect_emotion_level(content)
        prev_score      = data.get("prev_emotion_score", 0.0)  # Orchestrator 透传上轮加权分
        weighted_score  = round(current_emotion["score"] * 0.6 + prev_score * 0.4, 1)

        # 用加权分重新定级
        if weighted_score >= 5:
            weighted_level = "high"
        elif weighted_score >= 2:
            weighted_level = "medium"
        else:
            weighted_level = "low"

        emotion_result = {
            **current_emotion,
            "score":          weighted_score,   # 覆盖为加权分
            "level":          weighted_level,   # 覆盖为加权后等级
            "raw_score":      current_emotion["score"],   # 保留本轮原始分供 debug
            "prev_score":     prev_score,
        }

        return AgentResponse(
            success=True,
            message=f"识别到意图: {self._get_intent_desc(best_intent)}",
            data={
                "intent":           best_intent.value,
                "confidence":       best_score,
                "all_scores":       {k.value: v for k, v in intent_scores.items()},
                "extracted_data":   extracted_data,
                "emotion_level":    emotion_result,
                "original_content": content,
            },
            next_agent=self._get_target_agent(best_intent),
        )

    def _analyze_intent(self, content: str, history: list = None) -> Dict[IntentType, float]:
        scores = {intent: 0.0 for intent in IntentType}
        content_lower = content.lower()
        for intent, cfg in self.intent_patterns.items():
            score = 0.0
            for kw in cfg["keywords"]:
                if kw in content_lower:
                    score += 1.0
                    if content_lower.find(kw) < len(content) // 3:
                        score += 0.5
            for pat in cfg["patterns"]:
                if re.search(pat, content_lower):
                    score += 2.0
            scores[intent] = score

        # ── 历史窗口辅助：上轮已命中的意图加权 0.3，避免短句切换误路由 ──
        if history:
            for hist_content in history:
                hist_lower = hist_content.lower()
                for intent, cfg in self.intent_patterns.items():
                    hist_hit = any(kw in hist_lower for kw in cfg["keywords"])
                    if hist_hit:
                        scores[intent] += 0.3   # 历史权重低于当前轮，仅辅助

        if max(scores.values()) < 0.5:
            scores[IntentType.UNKNOWN] = 1.0
        return scores

    def _extract_entities(self, content: str, intent: IntentType) -> Dict[str, str]:
        data = {}
        # 订单号 - 优先匹配 "订单号XXXXXXXX" 格式，再匹配裸数字
        for pat in [r"订单[号编号]?\s*(\d{10,20})", r"\b(20\d{12,14})\b", r"\b(\d{10,20})\b"]:
            m = re.search(pat, content)
            if m:
                data["order_id"] = m.group(1)
                break
        # 快递单号
        m = re.search(r"\b([A-Z]{2}\d{9,13})\b", content, re.IGNORECASE)
        if m:
            data["tracking_number"] = m.group(1)
        # 手机号
        m = re.search(r"(1[3-9]\d{9})", content)
        if m:
            data["phone"] = m.group(1)
        if intent == IntentType.REFUND:
            if any(kw in content for kw in ["质量", "坏", "破", "瑕疵"]):
                data["refund_reason"] = "质量问题"
            elif any(kw in content for kw in ["不喜欢", "不合适", "不想要", "七天无理由"]):
                data["refund_reason"] = "七天无理由"
            elif any(kw in content for kw in ["不符", "不一样", "描述"]):
                data["refund_reason"] = "描述不符"
        return data

    def _detect_emotion_level(self, content: str) -> Dict:
        scores = {"angry": 0, "urgent": 0, "disappointed": 0}
        for etype, kws in self.emotion_keywords.items():
            for kw in kws:
                scores[etype] += content.count(kw)
        total = sum(scores.values())
        level = "high" if total >= 5 else "medium" if total >= 2 else "low"
        return {
            "level": level,
            "score": total,          # ← 修复：补充综合分数字段，之前缺失导致一直取到 0
            "scores": scores,
            "dominant": max(scores, key=scores.get) if total > 0 else None,
        }

    def _get_target_agent(self, intent: IntentType) -> str:
        return {
            IntentType.ORDER: "order",
            IntentType.LOGISTICS: "logistics",
            IntentType.REFUND: "refund",
            IntentType.COMPLAINT: "complaint",
            IntentType.UNKNOWN: "order",
        }.get(intent, "order")

    def _get_intent_desc(self, intent: IntentType) -> str:
        return {
            IntentType.ORDER: "订单问题",
            IntentType.LOGISTICS: "物流查询",
            IntentType.REFUND: "退款售后",
            IntentType.COMPLAINT: "投诉建议",
            IntentType.UNKNOWN: "其他问题",
        }.get(intent, "未知")
