"""
智能客服工作台 — Streamlit Demo
布局：左侧对话区 + 右侧分析面板（意图/实体/情绪/知识库）
"""
import io
import sys
import contextlib
import streamlit as st

sys.path.insert(0, ".")
from orchestrator import get_orchestrator
from knowledge.faq_store import FAQ_DOCUMENTS

# ── 页面配置 ─────────────────────────────────────────────
st.set_page_config(
    page_title="智能客服工作台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 样式 ─────────────────────────────────────────────────
st.markdown("""
<style>
/* 整体背景 */
.stApp { background: #f0f2f6; }

/* 顶部 Banner */
.banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.banner-title {
    color: #fff;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.5px;
}
.banner-sub {
    color: #a0b4d0;
    font-size: 13px;
    margin: 4px 0 0;
}

/* 聊天区卡片 */
.chat-card {
    background: #fff;
    border-radius: 12px;
    padding: 20px;
    min-height: 520px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}

/* 消息气泡 */
.msg-user {
    background: #0f3460;
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 6px 0 6px auto;
    max-width: 78%;
    width: fit-content;
    font-size: 14px;
    line-height: 1.5;
}
.msg-bot {
    background: #f1f5f9;
    color: #1e293b;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    margin: 6px auto 6px 0;
    max-width: 82%;
    width: fit-content;
    font-size: 14px;
    line-height: 1.5;
    border-left: 3px solid #0f3460;
}
.msg-label-user {
    text-align: right;
    font-size: 11px;
    color: #94a3b8;
    margin-bottom: 2px;
}
.msg-label-bot {
    text-align: left;
    font-size: 11px;
    color: #94a3b8;
    margin-bottom: 2px;
}
.escalate-badge {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    color: #92400e;
    margin-top: 4px;
}

/* 分析面板卡片 */
.panel-card {
    background: #fff;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
.panel-title {
    font-size: 12px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

/* 意图徽章 */
.intent-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
}

/* 实体标签 */
.entity-row {
    display: flex;
    align-items: center;
    margin: 6px 0;
    gap: 8px;
}
.entity-key {
    font-size: 12px;
    color: #64748b;
    width: 72px;
    flex-shrink: 0;
}
.entity-val {
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 13px;
    font-weight: 500;
    font-family: monospace;
}

/* 情绪指示器 */
.emotion-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
}

/* 知识库条目 */
.kb-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
}
.kb-q { font-size: 13px; font-weight: 600; color: #1e293b; margin-bottom: 4px; }
.kb-a { font-size: 12px; color: #475569; line-height: 1.6; }
.kb-tag {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 11px;
    margin-bottom: 6px;
}
.kb-hit {
    border-color: #0f3460;
    background: #eff6ff;
}

/* 统计小卡片 */
.stat-row {
    display: flex;
    gap: 8px;
    margin-top: 4px;
}
.stat-box {
    flex: 1;
    background: #f1f5f9;
    border-radius: 8px;
    padding: 8px;
    text-align: center;
}
.stat-num { font-size: 20px; font-weight: 700; color: #0f3460; }
.stat-lbl { font-size: 11px; color: #64748b; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Session State 初始化 ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []        # [{role, content, escalate, reason}]
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None      # 最新一轮分析结果
if "orch" not in st.session_state:
    st.session_state.orch = get_orchestrator()
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "escalate_count" not in st.session_state:
    st.session_state.escalate_count = 0


# ── 辅助函数 ─────────────────────────────────────────────

INTENT_STYLE = {
    "order":     ("#dbeafe", "#1d4ed8", "📦 订单"),
    "logistics": ("#dcfce7", "#15803d", "🚚 物流"),
    "refund":    ("#ffedd5", "#c2410c", "💰 退款"),
    "complaint": ("#fee2e2", "#b91c1c", "😤 投诉"),
    "unknown":   ("#f1f5f9", "#475569", "❓ 未知"),
}

EMOTION_STYLE = {
    "low":    ("#dcfce7", "#15803d", "😊 平稳"),
    "medium": ("#fef9c3", "#a16207", "😤 中等"),
    "high":   ("#fee2e2", "#b91c1c", "😡 激动"),
}

ENTITY_LABELS = {
    "order_id":        "订单号",
    "tracking_number": "快递单号",
    "phone":           "手机号",
    "refund_reason":   "退款原因",
}

def get_kb_for_intent(intent: str, rag_sources: list) -> list:
    """返回与意图匹配的 FAQ 条目，RAG 命中的排在前面。"""
    cat_map = {"refund": "refund", "logistics": "logistics",
               "order": None, "complaint": None, "unknown": None}
    cat = cat_map.get(intent)
    if not cat:
        return []
    docs = [d for d in FAQ_DOCUMENTS if d["category"] == cat]
    hit_ids = set(rag_sources)
    docs.sort(key=lambda d: (0 if d["id"] in hit_ids else 1))
    return docs[:4]

def send_message(user_input: str):
    """发送消息，捕获 orchestrator stdout，更新 session state。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = st.session_state.orch.process_message(
            user_input, st.session_state.session_id
        )
    st.session_state.session_id = result["session_id"]

    # 更新消息列表
    st.session_state.messages.append({
        "role": "user", "content": user_input,
        "escalate": False, "reason": "",
    })
    st.session_state.messages.append({
        "role": "bot", "content": result["response"],
        "escalate": result.get("need_escalate", False),
        "reason": result.get("escalate_reason", ""),
    })

    # 提取分析数据
    orch = st.session_state.orch
    sid  = st.session_state.session_id
    trace = orch.sessions[sid]["traces"][-1] if orch.sessions.get(sid, {}).get("traces") else {}
    ctx   = trace.get("context_snapshot", {})

    st.session_state.analysis = {
        "intent":      result.get("intent", "unknown"),
        "confidence":  trace.get("router_confidence", 0),
        "emotion_score": trace.get("router_emotion_score", 0),
        "emotion_level": trace.get("router_emotion_level", "low"),
        "entities":    {k: ctx.get(k) for k in ENTITY_LABELS if ctx.get(k)},
        "rag_used":    result.get("rag_used", False),
        "rag_sources": result.get("rag_sources", []),
    }
    st.session_state.turn_count += 1
    if result.get("need_escalate"):
        st.session_state.escalate_count += 1


# ── Banner ───────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <div>
    <p class="banner-title">🤖 智能客服工作台</p>
    <p class="banner-sub">RouterAgent → Order / Logistics / Refund / Complaint · 实时意图识别 · RAG 知识检索</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 主布局：左 60% 对话 + 右 40% 分析 ───────────────────
col_chat, col_panel = st.columns([6, 4], gap="medium")

# ════════════════════════════════════════
# 左栏：对话区
# ════════════════════════════════════════
with col_chat:
    # 顶部工具栏
    tc1, tc2, tc3 = st.columns([4, 2, 2])
    with tc2:
        if st.button("🔄 新会话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.analysis = None
            st.session_state.turn_count = 0
            st.session_state.escalate_count = 0
            st.rerun()
    with tc3:
        total = st.session_state.turn_count
        esc   = st.session_state.escalate_count
        rate  = f"{esc/total*100:.0f}%" if total else "—"
        st.markdown(
            f"<div style='text-align:right;font-size:12px;color:#64748b;padding-top:8px'>"
            f"↗ 升级率 <b>{rate}</b></div>",
            unsafe_allow_html=True
        )

    # 消息历史
    chat_html = '<div class="chat-card">'
    if not st.session_state.messages:
        chat_html += """
        <div style="text-align:center;padding:80px 20px;color:#94a3b8;">
          <div style="font-size:48px;margin-bottom:16px">💬</div>
          <div style="font-size:16px;font-weight:600;margin-bottom:8px">开始您的对话</div>
          <div style="font-size:13px">支持：查订单 / 查物流 / 退款 / 投诉</div>
        </div>"""
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="msg-label-user">👤 用户</div>
                <div class="msg-user">{msg["content"]}</div>"""
            else:
                content = msg["content"].replace("\n", "<br>")
                chat_html += f"""
                <div class="msg-label-bot">🤖 智能客服</div>
                <div class="msg-bot">{content}</div>"""
                if msg.get("escalate"):
                    chat_html += f"""
                    <div class="escalate-badge">
                      ⚠️ 已触发人工升级 — {msg.get("reason", "")}
                    </div>"""
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # 输入框
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        fc1, fc2 = st.columns([8, 2])
        with fc1:
            user_input = st.text_input(
                label="input",
                placeholder="输入您的问题，例如：查订单 / 我要退款 / 快递到哪了…",
                label_visibility="collapsed",
            )
        with fc2:
            submitted = st.form_submit_button("发送 ▶", use_container_width=True)

    if submitted and user_input.strip():
        send_message(user_input.strip())
        st.rerun()

    # 快捷语
    st.markdown("<div style='font-size:12px;color:#94a3b8;margin-top:4px'>快捷测试：</div>",
                unsafe_allow_html=True)
    shortcuts = [
        "查订单 202404150002",
        "SF1234567890 到哪了",
        "我要退款，质量问题",
        "你们服务太差了",
        "气死了！找经理",
    ]
    sc_cols = st.columns(len(shortcuts))
    for col, s in zip(sc_cols, shortcuts):
        with col:
            if st.button(s, key=f"sc_{s}", use_container_width=True):
                send_message(s)
                st.rerun()

# ════════════════════════════════════════
# 右栏：分析面板
# ════════════════════════════════════════
with col_panel:

    # ── 空状态 ──────────────────────────
    if st.session_state.analysis is None:
        st.markdown("""
        <div class="panel-card" style="text-align:center;padding:60px 20px;color:#94a3b8;">
          <div style="font-size:40px;margin-bottom:12px">📊</div>
          <div style="font-size:14px;font-weight:600">等待对话</div>
          <div style="font-size:12px;margin-top:6px">发送消息后，意图识别与实体分析将在此实时呈现</div>
        </div>""", unsafe_allow_html=True)
    else:
        a = st.session_state.analysis

        # ── 意图 & 情绪 ─────────────────
        intent = a["intent"]
        bg, fg, label = INTENT_STYLE.get(intent, INTENT_STYLE["unknown"])
        conf   = min(a["confidence"] / 10.0, 1.0)

        ebg, efg, elabel = EMOTION_STYLE.get(a["emotion_level"], EMOTION_STYLE["low"])
        escore = a["emotion_score"]

        st.markdown(f"""
        <div class="panel-card">
          <div class="panel-title">🧭 意图识别</div>
          <span class="intent-badge" style="background:{bg};color:{fg}">{label}</span>
          <div style="margin:12px 0 4px;font-size:12px;color:#64748b">
            置信度 &nbsp;<b style="color:#0f3460">{a['confidence']:.1f}</b>
          </div>
          <div style="background:#e2e8f0;border-radius:6px;height:8px;overflow:hidden">
            <div style="background:{fg};width:{conf*100:.0f}%;height:100%;border-radius:6px;transition:width .4s"></div>
          </div>
          <div style="margin-top:14px;font-size:12px;color:#64748b">情绪状态</div>
          <div style="margin-top:6px">
            <span class="emotion-pill" style="background:{ebg};color:{efg}">{elabel}</span>
            <span style="font-size:12px;color:#94a3b8;margin-left:8px">分值 {escore:.1f}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── 实体识别 ─────────────────────
        entities = a["entities"]
        entity_html = '<div class="panel-card"><div class="panel-title">🏷️ 实体识别</div>'
        if entities:
            for k, v in entities.items():
                lbl = ENTITY_LABELS.get(k, k)
                entity_html += f"""
                <div class="entity-row">
                  <span class="entity-key">{lbl}</span>
                  <span class="entity-val">{v}</span>
                </div>"""
        else:
            entity_html += '<div style="font-size:13px;color:#94a3b8;padding:8px 0">本轮暂未识别到实体</div>'
        entity_html += "</div>"
        st.markdown(entity_html, unsafe_allow_html=True)

        # ── 知识库 ───────────────────────
        kb_docs = get_kb_for_intent(intent, a["rag_sources"])
        kb_html = '<div class="panel-card"><div class="panel-title">📚 相关知识库</div>'

        if a["rag_used"] and a["rag_sources"]:
            kb_html += f'<div style="font-size:12px;color:#15803d;margin-bottom:8px">✅ RAG 检索命中 {len(a["rag_sources"])} 条</div>'
        elif kb_docs:
            kb_html += '<div style="font-size:12px;color:#94a3b8;margin-bottom:8px">按意图匹配（未触发向量检索）</div>'

        if kb_docs:
            for doc in kb_docs:
                is_hit = doc["id"] in a["rag_sources"]
                hit_cls = "kb-hit" if is_hit else ""
                hit_icon = "🔍 " if is_hit else ""
                answer = doc["answer"].replace("\n", "<br>")
                kb_html += f"""
                <div class="kb-item {hit_cls}">
                  <div class="kb-tag">{doc['category'].upper()} · {doc['id']}</div>
                  <div class="kb-q">{hit_icon}{doc['question']}</div>
                  <div class="kb-a">{answer}</div>
                </div>"""
        else:
            kb_html += '<div style="font-size:13px;color:#94a3b8;padding:8px 0">当前意图暂无匹配知识库条目</div>'
        kb_html += "</div>"
        st.markdown(kb_html, unsafe_allow_html=True)

    # ── 会话统计 ──────────────────────────
    total = st.session_state.turn_count
    esc   = st.session_state.escalate_count
    stats = st.session_state.orch.get_stats() if total > 0 else {}
    rag_h = stats.get("rag_hits", 0)

    st.markdown(f"""
    <div class="panel-card">
      <div class="panel-title">📈 会话统计</div>
      <div class="stat-row">
        <div class="stat-box"><div class="stat-num">{total}</div><div class="stat-lbl">总轮次</div></div>
        <div class="stat-box"><div class="stat-num">{esc}</div><div class="stat-lbl">已升级</div></div>
        <div class="stat-box"><div class="stat-num">{rag_h}</div><div class="stat-lbl">RAG命中</div></div>
      </div>
    </div>""", unsafe_allow_html=True)
