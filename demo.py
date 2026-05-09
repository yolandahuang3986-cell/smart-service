#!/usr/bin/env python3
"""
智能客服多Agent系统 - 交互式 Demo

用法：
    python3 demo.py           # 交互模式
    python3 demo.py --test    # 自动测试用例
"""
import sys
import argparse
from orchestrator import get_orchestrator


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🤖  智能客服多 Agent 系统  Demo                      ║
║                                                              ║
║   RouterAgent → Order / Logistics / Refund / Complaint      ║
╚══════════════════════════════════════════════════════════════╝

支持的对话类型：
  📦 订单  ："查订单"、"改地址"、"取消订单"
  🚚 物流  ："快递到哪了"、"查物流 SF1234567890"
  💰 退款  ："我要退款"、"七天无理由退货"
  😤 投诉  ："太差了"、"我要投诉"、"找经理"

输入 quit / exit 退出   输入 stats 查看统计   输入 new 开新会话
"""


def interactive():
    print(BANNER)
    orch = get_orchestrator()
    session_id = None
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n感谢使用，再见！👋")
                break
            if user_input.lower() == "stats":
                print_stats(orch)
                continue
            if user_input.lower() == "new":
                session_id = None
                print("✨ 已创建新会话")
                continue
            result = orch.process_message(user_input, session_id)
            session_id = result["session_id"]
            print(f"\n🤖 客服: {result['response']}")
            if result.get("need_escalate"):
                print(f"\n⚠️  系统提示：已触发人工升级 — {result.get('escalate_reason')}")
        except KeyboardInterrupt:
            print("\n\n感谢使用，再见！👋")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def print_stats(orch):
    s = orch.get_stats()
    rate = (s["escalation_count"] / s["total_requests"] * 100) if s["total_requests"] else 0
    print(f"\n{'='*50}")
    print("📊 系统统计")
    print(f"{'='*50}")
    print(f"总请求数   : {s['total_requests']}")
    print(f"总会话数   : {s['total_sessions']}")
    print(f"活跃会话   : {s['active_sessions']}")
    print(f"已升级人工 : {s['escalated_sessions']}")
    print(f"升级率     : {rate:.1f}%")
    print("\n意图分布：")
    for intent, cnt in s["intent_distribution"].items():
        if cnt:
            print(f"  • {intent}: {cnt}")
    print("="*50)


def run_tests():
    print(f"\n{'='*60}")
    print("🧪 自动化测试用例")
    print(f"{'='*60}\n")
    orch = get_orchestrator()

    cases = [
        ("我想查一下我的订单",                        "订单查询-无单号"),
        ("订单号202404150002改地址",                   "修改地址"),
        ("取消订单202404150002",                      "取消订单"),
        ("SF1234567890快递到哪了",                    "物流查询"),
        ("查一下我的快递",                            "物流查询-无单号"),
        ("订单202404160001我要退款，质量问题",          "退款-质量问题"),
        ("我要退货，七天无理由",                       "退款-七天无理由"),
        ("你们服务太差了，我要投诉",                   "投诉-一般不满"),
        ("气死我了！我要曝光你们！",                   "投诉-情绪激动"),
        ("找你们经理来，不要机器人",                   "投诉-要求人工"),
    ]

    session_id = None
    for i, (inp, desc) in enumerate(cases, 1):
        print(f"\n{'─'*60}")
        print(f"测试 {i:02d}/{len(cases)}：{desc}")
        print(f"{'─'*60}")
        result = orch.process_message(inp, session_id)
        session_id = result["session_id"]
        print(f"\n🤖 客服回复:\n{result['response']}")
        if result.get("need_escalate"):
            print(f"\n⚠️  已升级人工：{result.get('escalate_reason')}")

    print(f"\n{'='*60}")
    print("✅ 测试完成")
    print(f"{'='*60}")
    print_stats(orch)


def main():
    parser = argparse.ArgumentParser(description="智能客服多Agent系统 Demo")
    parser.add_argument("--test", action="store_true", help="运行自动化测试用例")
    args = parser.parse_args()
    if args.test:
        run_tests()
    else:
        interactive()


if __name__ == "__main__":
    main()
