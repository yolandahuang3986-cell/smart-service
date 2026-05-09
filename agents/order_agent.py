"""
订单Agent - 处理订单查询、修改地址、取消订单
"""
import re
from typing import Dict, Optional
from datetime import datetime, timedelta
from .base_agent import BaseAgent, Message, AgentResponse, IntentType


class OrderAgent(BaseAgent):

    def __init__(self, guardrail=None):
        super().__init__("order", "OrderAgent")
        self._guardrail = guardrail
        self.mock_orders = self._init_mock_orders()

    def _init_mock_orders(self) -> Dict[str, Dict]:
        t = datetime.now()
        return {
            "202404160001": {
                "order_id": "202404160001",
                "status": "已发货", "status_code": "shipped",
                "product_name": "iPhone 15 Pro Max", "product_price": 9999.00,
                "quantity": 1, "total_amount": 9999.00,
                "create_time": (t - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"),
                "ship_time": (t - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "receive_address": {
                    "name": "张三", "phone": "138****8888",
                    "province": "广东省", "city": "深圳市",
                    "district": "南山区", "detail": "科技园南区XX栋XX室",
                },
                "can_modify_address": False, "can_cancel": False,
                "tracking_number": "SF1234567890",
            },
            "202404150002": {
                "order_id": "202404150002",
                "status": "待发货", "status_code": "pending_ship",
                "product_name": "AirPods Pro 2", "product_price": 1899.00,
                "quantity": 2, "total_amount": 3798.00,
                "create_time": (t - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "ship_time": None,
                "receive_address": {
                    "name": "张三", "phone": "138****8888",
                    "province": "广东省", "city": "深圳市",
                    "district": "福田区", "detail": "华强北路XX号XX室",
                },
                "can_modify_address": True, "can_cancel": True,
                "tracking_number": None,
            },
            "202404100003": {
                "order_id": "202404100003",
                "status": "已完成", "status_code": "completed",
                "product_name": "iPad Air 5", "product_price": 4799.00,
                "quantity": 1, "total_amount": 4799.00,
                "create_time": (t - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"),
                "ship_time": (t - timedelta(days=9)).strftime("%Y-%m-%d %H:%M:%S"),
                "receive_address": {
                    "name": "张三", "phone": "138****8888",
                    "province": "广东省", "city": "深圳市",
                    "district": "罗湖区", "detail": "人民南路XX号XX室",
                },
                "can_modify_address": False, "can_cancel": False,
                "tracking_number": "SF0987654321",
            },
        }

    def process(self, message: Message) -> AgentResponse:
        content = message.content
        data = message.data
        self.log(f"处理订单请求: {content}")

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
        if any(kw in content for kw in ["改地址", "换地址", "修改地址"]):
            return self._handle_address_change(order_id, content)
        elif any(kw in content for kw in ["取消", "不要了"]):
            return self._handle_cancel_order(order_id, content)
        elif any(kw in content for kw in ["所有", "全部", "列表"]):
            return self._handle_list_orders()
        else:
            return self._handle_query_order(order_id, content)

    # ---------- query ----------
    def _handle_query_order(self, order_id: Optional[str], content: str) -> AgentResponse:
        if order_id and order_id in self.mock_orders:
            order = self.mock_orders[order_id]
            return AgentResponse(success=True, message=self._format_order_info(order),
                                 data={"order": order, "action": "query"})
        elif order_id:
            return AgentResponse(success=False,
                                 message=f"未找到订单号 {order_id} 的信息，请确认订单号是否正确。",
                                 data={"action": "query", "order_id": order_id})
        else:
            recent = list(self.mock_orders.values())[:2]
            msg = "未指定订单号，为您查询最近的订单：\n\n"
            for o in recent:
                msg += self._format_order_brief(o) + "\n---\n"
            msg += "\n如需查询特定订单，请提供订单号。"
            return AgentResponse(success=True, message=msg,
                                 data={"orders": recent, "action": "list_recent"})

    # ---------- address ----------
    def _handle_address_change(self, order_id: Optional[str], content: str) -> AgentResponse:
        if not order_id:
            m = re.search(r"订单[号编号]?\s*(\d{10,20})", content)
            if m:
                order_id = m.group(1)
        if not order_id:
            return AgentResponse(success=False,
                                 message="修改地址需要提供订单号，请提供您要修改的订单号。",
                                 data={"action": "change_address", "need_info": "order_id"})
        if order_id not in self.mock_orders:
            return AgentResponse(success=False,
                                 message=f"未找到订单号 {order_id}，请确认是否正确。",
                                 data={"action": "change_address", "order_id": order_id})
        order = self.mock_orders[order_id]
        if not order["can_modify_address"]:
            return AgentResponse(success=False,
                                 message=f"抱歉，订单 {order_id} 当前状态为【{order['status']}】，已无法修改地址。",
                                 data={"order": order, "action": "change_address", "can_modify": False})
        current_addr = self._format_address(order["receive_address"])
        return AgentResponse(success=True,
                             message=f"订单 {order_id} 可以修改地址。\n\n"
                                     f"当前地址：{current_addr}\n\n"
                                     f"请提供新的收货地址（省市区+详细地址+姓名+电话）。",
                             data={"order": order, "action": "change_address", "need_info": "new_address"})

    # ---------- cancel ----------
    def _handle_cancel_order(self, order_id: Optional[str], content: str) -> AgentResponse:
        if not order_id:
            m = re.search(r"\b(\d{10,20})\b", content)
            if m:
                order_id = m.group(1)
        if not order_id:
            return AgentResponse(success=False,
                                 message="取消订单需要提供订单号，请提供您要取消的订单号。",
                                 data={"action": "cancel", "need_info": "order_id"})
        if order_id not in self.mock_orders:
            return AgentResponse(success=False,
                                 message=f"未找到订单号 {order_id}，请确认是否正确。",
                                 data={"action": "cancel", "order_id": order_id})
        order = self.mock_orders[order_id]
        if not order["can_cancel"]:
            return AgentResponse(success=False,
                                 message=f"抱歉，订单 {order_id} 当前状态为【{order['status']}】，已无法取消。",
                                 data={"order": order, "action": "cancel", "can_cancel": False})
        order.update({"status": "已取消", "status_code": "cancelled",
                      "can_cancel": False, "can_modify_address": False,
                      "cancel_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        return AgentResponse(success=True,
                             message=f"✅ 订单 {order_id} 已成功取消！\n\n"
                                     f"商品：{order['product_name']} x{order['quantity']}\n"
                                     f"金额：¥{order['total_amount']:.2f}\n\n"
                                     f"退款将在 3-7 个工作日内原路退回您的支付账户。",
                             data={"order": order, "action": "cancel"})

    # ---------- list ----------
    def _handle_list_orders(self) -> AgentResponse:
        orders = list(self.mock_orders.values())
        msg = f"您共有 {len(orders)} 个订单：\n\n"
        for o in orders:
            msg += self._format_order_brief(o) + "\n---\n"
        return AgentResponse(success=True, message=msg, data={"orders": orders, "action": "list_all"})

    # ---------- helpers ----------
    _STATUS_EMOJI = {"待发货": "📦", "已发货": "🚚", "已完成": "✅", "已取消": "❌"}

    def _format_order_info(self, o: Dict) -> str:
        emoji = self._STATUS_EMOJI.get(o["status"], "📋")
        info = (f"{emoji} 订单详情\n"
                f"{'━'*20}\n"
                f"📋 订单号：{o['order_id']}\n"
                f"📌 状态：{o['status']}\n"
                f"📅 下单时间：{o['create_time']}\n\n"
                f"🛒 商品：{o['product_name']} x{o['quantity']}  ¥{o['total_amount']:.2f}\n\n"
                f"📍 收货地址：{self._format_address(o['receive_address'])}")
        if o.get("tracking_number"):
            info += f"\n🚚 物流单号：{o['tracking_number']}"
        actions = []
        if o.get("can_modify_address"):
            actions.append("可修改地址")
        if o.get("can_cancel"):
            actions.append("可取消订单")
        if actions:
            info += "\n💡 " + " | ".join(actions)
        return info

    def _format_order_brief(self, o: Dict) -> str:
        emoji = self._STATUS_EMOJI.get(o["status"], "📋")
        return (f"{emoji} {o['order_id']} | {o['status']}\n"
                f"   {o['product_name']} x{o['quantity']} | ¥{o['total_amount']:.2f}")

    def _format_address(self, a: Dict) -> str:
        return f"{a['province']}{a['city']}{a['district']}{a['detail']} ({a['name']} {a['phone']})"
