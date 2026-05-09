"""
退款Agent - 判断是否可退、计算退款金额、生成退款方案
"""
import re
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from .base_agent import BaseAgent, Message, AgentResponse, IntentType


class RefundReason(Enum):
    QUALITY_ISSUE = "quality_issue"
    WRONG_ITEM = "wrong_item"
    NOT_AS_DESCRIBED = "not_as_described"
    SEVEN_DAY_NO_REASON = "seven_day"
    DAMAGED = "damaged"
    LATE_DELIVERY = "late_delivery"
    OTHER = "other"


class RefundAgent(BaseAgent):

    def __init__(self, retriever=None, guardrail=None):
        super().__init__("refund", "RefundAgent", retriever=retriever)
        self._guardrail = guardrail
        self.mock_orders = self._init_mock_orders()
        self.refund_records: Dict[str, Dict] = {}

    def _init_mock_orders(self) -> Dict[str, Dict]:
        t = datetime.now()
        return {
            "202404160001": {
                "order_id": "202404160001", "status": "已发货",
                "product_name": "iPhone 15 Pro Max", "product_price": 9999.00,
                "quantity": 1, "total_amount": 9999.00, "shipping_fee": 0.0,
                "create_time": (t - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"),
                "receive_time": None, "can_refund": True,
                "refund_deadline": (t + timedelta(days=4)).strftime("%Y-%m-%d"),
            },
            "202404100003": {
                "order_id": "202404100003", "status": "已完成",
                "product_name": "iPad Air 5", "product_price": 4799.00,
                "quantity": 1, "total_amount": 4799.00, "shipping_fee": 0.0,
                "create_time": (t - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"),
                "receive_time": (t - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S"),
                "can_refund": True,
                "refund_deadline": (t + timedelta(days=1)).strftime("%Y-%m-%d"),
            },
            "202404010004": {
                "order_id": "202404010004", "status": "已完成",
                "product_name": "MacBook Pro", "product_price": 14999.00,
                "quantity": 1, "total_amount": 14999.00, "shipping_fee": 0.0,
                "create_time": (t - timedelta(days=20)).strftime("%Y-%m-%d %H:%M:%S"),
                "receive_time": (t - timedelta(days=16)).strftime("%Y-%m-%d %H:%M:%S"),
                "can_refund": False,
                "refund_deadline": (t - timedelta(days=9)).strftime("%Y-%m-%d"),
            },
        }

    def process(self, message: Message) -> AgentResponse:
        content = message.content
        data = message.data
        self.log(f"处理退款请求: {content}")

        # ── Layer2：敏感操作拦截 ──────────────────────────
        if self._guardrail:
            emotion_score = data.get("emotion_level", {}).get("score", 0)
            block_msg = self._guardrail.check_sensitive_operation(content, emotion_score)
            if block_msg:
                return AgentResponse(
                    success=False,
                    message=block_msg,
                    data={"action": "sensitive_blocked"},
                )

        order_id = data.get("extracted_data", {}).get("order_id")
        if not order_id:
            for pat in [r"订单[号编号]?\s*(\d{10,20})", r"\b(\d{10,20})\b"]:
                m = re.search(pat, content)
                if m:
                    order_id = m.group(1)
                    break
        refund_reason = self._extract_refund_reason(content, data)
        if not order_id:
            return AgentResponse(success=False,
                                 message="申请退款需要提供订单号，请告知您要退款的订单号。",
                                 data={"action": "apply", "need_info": "order_id"})
        if order_id not in self.mock_orders:
            return AgentResponse(success=False,
                                 message=f"未找到订单号 {order_id}，请确认是否正确。",
                                 data={"action": "apply", "order_id": order_id})
        order = self.mock_orders[order_id]
        can, reason = self._check_eligibility(order, refund_reason)
        if not can:
            return AgentResponse(success=False,
                                 message=f"抱歉，订单 {order_id} 暂时无法申请退款。\n\n原因：{reason}",
                                 data={"order": order, "action": "apply", "can_refund": False})
        amount = self._calc_amount(order, refund_reason)
        plan = self._gen_plan(order, amount, refund_reason)

        # ── Layer3：输出层模糊话术过滤 ────────────────────
        if self._guardrail:
            emotion_score = data.get("emotion_level", {}).get("score", 0)
            plan = self._guardrail.check_output(plan, emotion_score)

        return AgentResponse(success=True, message=plan,
                             data={"order": order, "action": "apply", "can_refund": True,
                                   "refund_amount": amount,
                                   "refund_reason": refund_reason.value if refund_reason else None})

    def _extract_refund_reason(self, content: str, data: Dict) -> Optional[RefundReason]:
        ext = data.get("extracted_data", {}).get("refund_reason", "")
        if ext == "质量问题":   return RefundReason.QUALITY_ISSUE
        if ext == "七天无理由": return RefundReason.SEVEN_DAY_NO_REASON
        if ext == "描述不符":   return RefundReason.NOT_AS_DESCRIBED
        c = content
        if any(kw in c for kw in ["质量", "坏", "破", "瑕疵", "故障"]):  return RefundReason.QUALITY_ISSUE
        if any(kw in c for kw in ["发错", "错发", "不是我要的"]):          return RefundReason.WRONG_ITEM
        if any(kw in c for kw in ["描述不符", "图文不符"]):                return RefundReason.NOT_AS_DESCRIBED
        if any(kw in c for kw in ["七天无理由", "不喜欢", "不合适"]):       return RefundReason.SEVEN_DAY_NO_REASON
        if any(kw in c for kw in ["破损", "碎了", "压坏"]):                return RefundReason.DAMAGED
        if any(kw in c for kw in ["没发货", "延迟", "等太久"]):             return RefundReason.LATE_DELIVERY
        return RefundReason.OTHER

    def _check_eligibility(self, order: Dict, reason: Optional[RefundReason], rag_results: list = None) -> Tuple[bool, str]:
        status = order.get("status", "")

        # ── 待发货：可直接取消并退款 ──────────────────────
        if status == "待发货":
            if order["order_id"] in self.refund_records:
                return False, "该订单已有退款申请正在处理中。"
            return True, "订单待发货，可直接取消并退款"

        # ── 已发货（配送中）：不可直接退款 ────────────────
        if status == "已发货":
            return False, (
                "订单当前正在配送中，暂时无法直接退款。\n\n"
                "您有两个选择：\n"
                "• 【拒收】联系快递员拒收，拒收确认后退款将在 3 个工作日内处理\n"
                "• 【等待签收】收到货后确认问题，再申请退货退款"
            )

        # ── 已完成/已签收：按原因和时限判断 ──────────────
        if reason == RefundReason.QUALITY_ISSUE:
            if order["order_id"] in self.refund_records:
                return False, "该订单已有退款申请正在处理中。"
            return True, "质量问题支持退款"

        if order.get("receive_time"):
            days = (datetime.now() - datetime.strptime(
                order["receive_time"], "%Y-%m-%d %H:%M:%S"
            )).days
            if days > 7:
                return False, (
                    f"订单已签收 {days} 天，超过七天无理由退货期限。\n"
                    "如有质量问题仍可申请退款，请注明具体问题。"
                )

        if order["order_id"] in self.refund_records:
            return False, "该订单已有退款申请正在处理中。"

        return True, "符合退款条件"

    def _calc_amount(self, order: Dict, reason: Optional[RefundReason]) -> Dict:
        total = order["total_amount"]
        ship = order.get("shipping_fee", 0)
        seller_fault = reason in [RefundReason.QUALITY_ISSUE, RefundReason.WRONG_ITEM,
                                  RefundReason.NOT_AS_DESCRIBED, RefundReason.DAMAGED]
        ship_refund = ship if seller_fault else 0
        product_refund = total - ship
        return {"product_amount": product_refund, "shipping_fee": ship_refund,
                "total": product_refund + ship_refund, "original_total": total,
                "responsibility": "seller" if seller_fault else "buyer"}

    def _gen_plan(self, order: Dict, amount: Dict, reason: Optional[RefundReason]) -> str:
        reason_text = {
            RefundReason.QUALITY_ISSUE: "质量问题",
            RefundReason.WRONG_ITEM: "发错货",
            RefundReason.NOT_AS_DESCRIBED: "描述不符",
            RefundReason.SEVEN_DAY_NO_REASON: "七天无理由退货",
            RefundReason.DAMAGED: "商品破损",
            RefundReason.LATE_DELIVERY: "未按时发货",
            RefundReason.OTHER: "协商退款",
        }.get(reason, "协商退款")
        msg = (f"✅ 退款申请评估通过\n\n"
               f"📋 订单信息\n{'━'*20}\n"
               f"订单号：{order['order_id']}\n"
               f"商品：{order['product_name']}\n"
               f"订单金额：¥{order['total_amount']:.2f}\n\n"
               f"💰 退款金额明细\n{'━'*20}\n"
               f"商品金额：¥{amount['product_amount']:.2f}\n"
               f"运费：¥{amount['shipping_fee']:.2f}\n"
               f"{'━'*20}\n"
               f"退款总额：¥{amount['total']:.2f}\n\n"
               f"📝 退款原因：{reason_text}\n")
        if amount["responsibility"] == "seller":
            msg += "💡 因卖家责任，运费全额退还\n"
        else:
            msg += "💡 七天无理由退货，运费由买家承担\n"
        msg += ("\n⏱️ 退款时效\n"
                "• 审核：1-3 个工作日\n"
                "• 到账：审核通过后 3-7 个工作日（原路退回）\n\n"
                "回复【确认退款】提交申请，回复【取消】放弃申请")
        return msg
