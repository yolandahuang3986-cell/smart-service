"""
物流Agent - 物流查询 + 模拟外部物流API调用
"""
import re
import time
import random
from typing import Dict, Optional
from datetime import datetime, timedelta
from .base_agent import BaseAgent, Message, AgentResponse, IntentType


class LogisticsAgent(BaseAgent):

    def __init__(self, retriever=None):
        super().__init__("logistics", "LogisticsAgent",retriever=retriever)
        self.carriers = {"SF": "顺丰速运", "JD": "京东物流",
                         "YT": "圆通速递", "ZT": "中通快递", "YD": "韵达速递"}
        self.mock_tracking = self._init_mock_tracking()

    def _init_mock_tracking(self) -> Dict[str, Dict]:
        t = datetime.now()
        return {
            "SF1234567890": {
                "tracking_number": "SF1234567890", "carrier_code": "SF",
                "carrier_name": "顺丰速运", "status": "派送中",
                "estimated_delivery": (t + timedelta(days=1)).strftime("%Y-%m-%d"),
                "timeline": [
                    {"time": (t - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
                     "status": "派送中", "detail": "快递员【王师傅 138****0000】正在派送中，请保持电话畅通",
                     "location": "北京市朝阳区"},
                    {"time": (t - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M"),
                     "status": "运输中", "detail": "快件已到达【北京朝阳营业点】",
                     "location": "北京市朝阳区"},
                    {"time": (t - timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
                     "status": "运输中", "detail": "快件已发往【北京朝阳营业点】",
                     "location": "深圳市"},
                    {"time": (t - timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
                     "status": "已揽收", "detail": "顺丰速运已收取快件",
                     "location": "深圳市南山区"},
                ],
            },
            "SF0987654321": {
                "tracking_number": "SF0987654321", "carrier_code": "SF",
                "carrier_name": "顺丰速运", "status": "已签收",
                "signed_time": (t - timedelta(days=3)).strftime("%Y-%m-%d %H:%M"),
                "signed_by": "本人",
                "timeline": [
                    {"time": (t - timedelta(days=3)).strftime("%Y-%m-%d %H:%M"),
                     "status": "已签收", "detail": "您的快件已签收，签收人：本人",
                     "location": "深圳市南山区"},
                    {"time": (t - timedelta(days=3, hours=-2)).strftime("%Y-%m-%d %H:%M"),
                     "status": "派送中", "detail": "快递员正在派送中",
                     "location": "深圳市南山区"},
                    {"time": (t - timedelta(days=4)).strftime("%Y-%m-%d %H:%M"),
                     "status": "运输中", "detail": "快件已到达【深圳南山营业点】",
                     "location": "深圳市南山区"},
                    {"time": (t - timedelta(days=5)).strftime("%Y-%m-%d %H:%M"),
                     "status": "已揽收", "detail": "顺丰速运已收取快件",
                     "location": "上海市"},
                ],
            },
        }

    def process(self, message: Message) -> AgentResponse:
        content = message.content
        data = message.data
        self.log(f"处理物流请求: {content}")
        tracking = data.get("extracted_data", {}).get("tracking_number")
        if not tracking:
            m = re.search(r"\b([A-Z]{2}\d{9,13})\b", content, re.IGNORECASE)
            if m:
                tracking = m.group(1)
        if tracking:
            return self._query_tracking(tracking)
        return AgentResponse(success=False,
                             message="请提供快递单号，我可以帮您查询物流状态。\n\n"
                                     "💡 获取单号方式：\n• 查看订单详情\n• 查看发货短信\n• 查看购物App物流页面",
                             data={"action": "query", "need_info": "tracking_number"})

    def _query_tracking(self, number: str) -> AgentResponse:
        self.log(f"查询物流单号: {number}")
        time.sleep(0.3)
        result, api_error = self._call_external_api(number)
        if api_error:
            code = number[:2].upper()
            carrier = self.carriers.get(code, "")
            hotline = {"SF": "顺丰客服 95338", "JD": "京东物流 950618",
                       "YT": "圆通客服 95554", "ZT": "中通客服 95311",
                       "YD": "韵达客服 95546"}.get(code, "快递公司官网")
            hint = f"您也可以直接联系{carrier}查询：{hotline}" if carrier else f"您也可以直接前往{hotline}查询"
            return AgentResponse(
                success=False,
                message=(f"抱歉，物流系统暂时无法查询，请稍后重试。\n\n"
                         f"单号：{number}\n"
                         f"💡 {hint}"),
                data={"action": "query", "tracking_number": number, "error": "api_unavailable"},
            )
        if result:
            return AgentResponse(success=True, message=self._format_tracking(result),
                                 data={"tracking": result, "action": "query", "source": "external_api"})
        return AgentResponse(
            success=False,
            message=(f"未找到快递单号 {number} 的物流信息。\n\n"
                     f"可能原因：\n• 单号有误\n• 刚发出尚未同步\n• 非合作快递公司"),
            data={"action": "query", "tracking_number": number, "error": "not_found"},
        )

    def _call_external_api(self, number: str) -> tuple[Optional[Dict], bool]:
        """
        调用外部物流 API。
        返回 (data, api_error)：
          - (Dict, False)：查询成功，有结果
          - (None, False)：查询成功，单号不存在
          - (None, True) ：API 不可用（超时 / 服务异常）
        """
        try:
            # 模拟 40% 概率服务异常
            if random.random() < 0.4:
                raise ConnectionError("外部物流 API 连接超时")
            # API 可用时，模拟单号存在与否
            if random.random() > 0.6:
                t = datetime.now()
                code = number[:2].upper()
                data = {
                    "tracking_number": number,
                    "carrier_code": code,
                    "carrier_name": self.carriers.get(code, "未知快递"),
                    "status": "运输中",
                    "estimated_delivery": (t + timedelta(days=2)).strftime("%Y-%m-%d"),
                    "timeline": [
                        {"time": (t - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"),
                         "status": "运输中", "detail": "快件已发往目的地", "location": "转运中心"},
                        {"time": (t - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M"),
                         "status": "已揽收", "detail": "快递员已揽收", "location": "发货地"},
                    ],
                }
                return data, False
            return None, False      # API 正常但单号不存在
        except Exception as e:
            self.log(f"外部 API 调用失败: {e}")
            return None, True       # API 本身不可用

    def _format_tracking(self, t: Dict) -> str:
        emoji = {"派送中": "🛵", "运输中": "🚚", "已揽收": "📦", "已签收": "✅"}.get(t["status"], "📋")
        info = (f"{emoji} 物流详情\n{'━'*20}\n"
                f"📋 快递单号：{t['tracking_number']}\n"
                f"🏢 快递公司：{t['carrier_name']}\n"
                f"📌 当前状态：{t['status']}\n")
        if t.get("estimated_delivery"):
            info += f"📅 预计送达：{t['estimated_delivery']}\n"
        if t.get("signed_time"):
            info += f"✍️ 签收时间：{t['signed_time']} (签收人：{t.get('signed_by','无')})\n"
        info += "\n📝 物流轨迹：\n"
        for i, ev in enumerate(t["timeline"]):
            prefix = "├─" if i < len(t["timeline"]) - 1 else "└─"
            info += f"\n{prefix} {ev['time']}\n   {ev['status']}：{ev['detail']}\n   📍 {ev['location']}"
        if t["status"] == "派送中":
            info += "\n\n💡 请保持电话畅通，快递员正在配送中。"
        elif t["status"] == "已签收":
            info += "\n\n💡 商品已签收，如有问题请及时申请售后。"
        return info
