# 🤖 智能客服多 Agent 系统

> 一个基于 Agent 架构的智能客服解决方案，支持订单、物流、退款、投诉四大业务场景的自动处理。

---

## 📁 项目结构

```
customer_service_agents/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py        # Agent 基类 + 消息协议
│   ├── router_agent.py      # 意图识别与路由
│   ├── order_agent.py       # 订单查询 / 改地址 / 取消
│   ├── logistics_agent.py   # 物流查询 + 模拟外部 API
│   ├── refund_agent.py      # 退款资格判断 + 金额计算
│   └── complaint_agent.py   # 情绪识别 + 安抚 + 人工升级
├── orchestrator.py          # 中央协调器 / 会话管理
├── demo.py                  # 交互式 Demo 入口
└── README.md
```

---

## 🏗️ 系统架构

```
用户输入
   │
   ▼
Orchestrator（协调器）
   │
   ├─→ RouterAgent（意图识别）
   │       │
   │       ├── ORDER     → OrderAgent
   │       ├── LOGISTICS → LogisticsAgent
   │       ├── REFUND    → RefundAgent
   │       └── COMPLAINT → ComplaintAgent
   │
   └─→ 返回结构化响应给用户
```

---

## 🚀 快速开始

```bash
# 进入项目目录
cd customer_service_agents

# 交互模式
python3 demo.py

# 自动化测试（10 个用例）
python3 demo.py --test
```

---

## 💬 支持的对话示例

| 用户输入 | 处理 Agent | 能力 |
|---------|-----------|------|
| 我想查一下我的订单 | OrderAgent | 返回最近订单列表 |
| 订单号202404150002改地址 | OrderAgent | 引导修改地址（待发货可改） |
| 取消订单202404150002 | OrderAgent | 取消并触发退款 |
| SF1234567890快递到哪了 | LogisticsAgent | 完整物流轨迹 |
| 订单202404160001退款，质量问题 | RefundAgent | 评估资格 + 退款方案 |
| 气死我了！我要曝光你们！ | ComplaintAgent | 情绪识别 → 升级人工 |
| 找经理，不要机器人 | ComplaintAgent | 明确要求 → 升级人工 |

---

## 🧠 各 Agent 详解

### RouterAgent — 意图路由

- 关键词 + 正则双层匹配，输出置信度分数
- 自动提取订单号、快递单号、手机号等实体
- 多维情绪检测（愤怒 / 急迫 / 失望）
- 路由到对应业务 Agent

### OrderAgent — 订单处理

- 查询订单详情 / 最近订单列表
- 修改收货地址（待发货状态）
- 取消订单（待发货状态），自动触发退款

### LogisticsAgent — 物流查询

- 查询完整物流轨迹时间线
- 内置顺丰、京东等快递数据
- 模拟调用外部物流 API（含随机成功/失败场景）

### RefundAgent — 退款处理

| 退款原因 | 运费承担 | 期限限制 |
|---------|---------|---------|
| 质量问题 | 卖家承担 | 无限制 |
| 发错货 | 卖家承担 | 无限制 |
| 七天无理由 | 买家承担 | 签收 7 天内 |
| 描述不符 | 卖家承担 | 无限制 |

### ComplaintAgent — 投诉与情绪

**情绪分级**：

| 等级 | 分数 | 处理方式 |
|------|------|---------|
| low | < 3 | 通用安抚话术 |
| medium | 3–7 | 针对性安抚 + 方案 |
| high | ≥ 8 | 升级人工 |

**自动升级人工的触发条件**：
- 情绪分数 ≥ 8
- 含"曝光/投诉/12315/法院"等威胁性词汇
- 明确要求"转人工/找经理"
- 同一会话重复投诉 ≥ 3 次

---

## 🔧 扩展指南

### 添加新业务 Agent

```python
# 1. 继承 BaseAgent
from agents import BaseAgent, Message, AgentResponse

class PaymentAgent(BaseAgent):
    def __init__(self):
        super().__init__("payment", "PaymentAgent")

    def process(self, message: Message) -> AgentResponse:
        # 业务逻辑
        return AgentResponse(success=True, message="处理结果", data={})

# 2. 在 orchestrator.py 中注册
self.payment_agent = PaymentAgent()
self.agents["payment"] = self.payment_agent

# 3. 在 RouterAgent 的 intent_patterns 中添加关键词
```

### 接入真实 API

各 Agent 中的 mock 数据可替换为真实 HTTP 调用：

```python
import requests

def _query_real_order(self, order_id: str):
    resp = requests.get(
        f"https://api.yourshop.com/orders/{order_id}",
        headers={"Authorization": f"Bearer {YOUR_TOKEN}"}
    )
    return resp.json()
```

---

## 📊 统计输出示例

```
📊 系统统计
==================================================
总请求数   : 10
总会话数   : 1
活跃会话   : 1
已升级人工 : 1
升级率     : 30.0%

意图分布：
  • order: 3
  • logistics: 2
  • refund: 2
  • complaint: 3
```

---

## 📄 License

MIT
