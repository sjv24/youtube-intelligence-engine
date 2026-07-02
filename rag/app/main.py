import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "rag"))

from pipeline.generator import buildGraph

import time
import html
import json
import uuid
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ENDURANCE // Interstellar Q&A",
    page_icon="🪐",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# MISSION CONTROL THEME — CSS
# ============================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --void: #04060c;
    --panel-solid: #0d1322;
    --line: #1c2742;
    --line-soft: #141b2c;
    --gold: #cf9f57;
    --gold-bright: #e8b86d;
    --amber-dim: #8a6a3a;
    --ice: #cfe0ff;
    --text: #e8ebf3;
    --muted: #8b93a8;
    --muted-2: #5c6478;
    --danger: #c6694f;
}

/* ---- base ---- */
html, body, [class*="css"] {
    background-color: var(--void) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-weight: 300;
}
.stApp {
    background: radial-gradient(ellipse at top, #0a0e1a 0%, var(--void) 60%);
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }

/* keep the sidebar re-open control visible + themed */
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    fill: var(--gold-bright) !important;
}
[data-testid="collapsedControl"] {
    visibility: visible !important;
}

/* live starfield canvas — injected into the page by the warp engine */
#mc-stars {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
}

/* dim + blur the whole app while traversing hyperspace */
.stApp { transition: filter .5s ease, transform .5s ease; }
.stApp.mc-warping { filter: blur(2px) brightness(.55); transform: scale(.99); }

/* keep the floating composer above the canvas */
div[data-testid="stBottom"] { z-index: 5 !important; }

/* collapse the zero-height script iframes so they never leave layout gaps */
div[data-testid="stElementContainer"]:has(iframe[height="0"]),
div.element-container:has(iframe[height="0"]) {
    display: none !important;
}

/* ---- mission header panel ---- */
.mc-header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 18px;
    padding: 18px 24px;
    background: rgba(13, 19, 34, 0.55);
    border: 1px solid var(--line);
    border-radius: 14px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 10px 40px -18px rgba(0, 0, 0, 0.85),
                inset 0 0 30px rgba(207, 159, 87, 0.04);
    margin-bottom: 18px;
}
.mc-brand { display: flex; align-items: center; gap: 16px; }
.mc-ring {
    width: 46px; height: 46px; border-radius: 50%;
    border: 1px solid rgba(207, 159, 87, 0.4);
    display: grid; place-items: center;
    position: relative; flex-shrink: 0;
}
.mc-ring::before {
    content: ''; position: absolute; inset: -5px; border-radius: 50%;
    border: 1px dashed rgba(207, 159, 87, 0.28);
    animation: mc-spin 14s linear infinite;
}
.mc-core {
    width: 14px; height: 14px; border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #fff, var(--gold-bright) 55%, transparent 80%);
    box-shadow: 0 0 14px var(--gold-bright), 0 0 34px rgba(207, 159, 87, 0.4);
    animation: mc-pulse 2.6s ease-in-out infinite;
}
@keyframes mc-spin { to { transform: rotate(360deg); } }
@keyframes mc-pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(0.75); opacity: 0.7; }
}
.mc-name {
    font-family: 'Oswald', sans-serif;
    font-weight: 600; font-size: 22px; letter-spacing: .3em;
    color: var(--text);
    text-shadow: 0 0 18px rgba(207, 159, 87, 0.35);
    line-height: 1.1;
}
.mc-name-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: .22em; text-transform: uppercase;
    color: var(--muted-2); margin-top: 5px;
}
.mc-telemetry {
    display: flex; gap: 26px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted-2); text-align: right;
}
.mc-telemetry b {
    display: block; font-size: 12px; font-weight: 500;
    color: var(--ice); letter-spacing: .08em; margin-top: 3px;
}
.mc-telemetry .gold b {
    color: var(--gold-bright);
    text-shadow: 0 0 10px rgba(207, 159, 87, 0.35);
}
@media (max-width: 700px) { .mc-telemetry { display: none; } }

.block-container { position: relative; z-index: 1; padding-top: 2.5rem; max-width: 760px; padding-bottom: 8rem; }

/* ---- eyebrow ---- */
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--gold);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.eyebrow::before {
    content: "―"; color: var(--gold);
}

/* ---- title ---- */
.mc-title {
    font-family: 'Oswald', sans-serif;
    font-weight: 600;
    letter-spacing: .04em;
    font-size: 38px;
    color: var(--text);
    margin: 0 0 6px 0;
    line-height: 1.15;
}
.mc-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--muted-2);
    border-top: 1px solid var(--line-soft);
    padding-top: 12px;
    margin-bottom: 18px;
}
.mc-lede {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 15px;
    line-height: 1.75;
    color: var(--muted);
    max-width: 620px;
    margin-bottom: 8px;
}

/* ---- divider ---- */
.hairline { border: none; border-top: 1px solid var(--line); margin: 34px 0; }

/* ---- FLOATING CHAT INPUT PILL ---- */
/* Kill EVERY grey wrapper Streamlit stacks behind the input, incl. the
   outer stBottom element the previous rules missed. */
div[data-testid="stBottom"],
div[data-testid="stBottom"] > div,
div[data-testid="stBottomBlockContainer"],
div[data-testid="stBottomBlockContainer"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Constrain + center the bottom block so the pill lines up with the page */
div[data-testid="stBottomBlockContainer"] {
    max-width: 760px !important;
    margin: 0 auto !important;
    padding-bottom: 22px !important;
}

/* The input is a self-contained floating composer box */
div[data-testid="stChatInput"] {
    background: rgba(13, 19, 34, 0.85) !important;
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    padding: 12px 14px 12px 20px !important;
    box-shadow: 0 10px 34px -12px rgba(0, 0, 0, 0.85),
                0 0 0 1px rgba(207, 159, 87, 0.06) inset !important;
    backdrop-filter: blur(6px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

/* Gold accent + lift when the capsule holds focus */
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 14px 40px -12px rgba(0, 0, 0, 0.9),
                0 0 18px -4px rgba(207, 159, 87, 0.35) !important;
}

/* Make the text field + its BaseWeb wrappers fill the whole composer width */
div[data-testid="stChatInput"] > div:first-child {
    flex: 1 1 auto !important;
    width: 100% !important;
    min-width: 0 !important;
    background: transparent !important;
}
div[data-testid="stChatInput"] [data-baseweb="textarea"],
div[data-testid="stChatInput"] [data-baseweb="base-input"] {
    width: 100% !important;
    background: transparent !important;
    border: none !important;
}

/* Textarea blends fully into the composer — no frame of its own */
div[data-testid="stChatInput"] textarea {
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0px !important;
    color: var(--ice) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 300 !important;
    padding: 4px 0px !important;
    min-height: 58px !important; /* gives the box its taller composer height */
    box-shadow: none !important;
}
div[data-testid="stChatInput"] textarea:focus {
    box-shadow: none !important;
}

/* Send button tuned to the void aesthetic */
div[data-testid="stChatInput"] button {
    background: transparent !important;
    color: var(--muted) !important;
    transition: color 0.3s ease;
}
div[data-testid="stChatInput"] button:hover {
    color: var(--gold-bright) !important;
}

/* ---- ARCHIVE (sidebar) ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090d18 0%, var(--void) 100%) !important;
    border-right: 1px solid var(--line-soft);
}

/* New Transmission — primary gold action */
.st-key-new_transmission button {
    background: rgba(207,159,87,.08) !important;
    border: 1px solid var(--amber-dim) !important;
    border-radius: 8px !important;
    color: var(--gold-bright) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: .10em !important;
    text-transform: uppercase !important;
    transition: all .25s ease;
}
.st-key-new_transmission button:hover {
    background: rgba(207,159,87,.16) !important;
    border-color: var(--gold) !important;
    color: var(--gold-bright) !important;
}

/* Session log rows */
[class*="st-key-session_"] button {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid var(--line) !important;
    border-radius: 0 !important;
    color: var(--muted) !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    transition: all .2s ease;
}
[class*="st-key-session_"] button:hover {
    background: rgba(10,14,24,.55) !important;
    border-left-color: var(--gold-bright) !important;
    color: var(--ice) !important;
}

/* Rename / delete / save icon buttons */
[class*="st-key-rename_"] button,
[class*="st-key-del_"] button {
    background: transparent !important;
    border: none !important;
    color: var(--muted-2) !important;
    padding: 8px 0 !important;
    min-height: 0 !important;
    transition: color .2s ease;
}
[class*="st-key-rename_"] button:hover { color: var(--gold-bright) !important; }
[class*="st-key-del_"] button:hover { color: var(--danger) !important; }

/* Per-row timestamp · turns caption */
.archive-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted-2);
    margin: -4px 0 14px 14px;
}

/* ---- status / routing chip ---- */
.routing-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--gold-bright);
    border: 1px solid var(--line);
    background: rgba(10,14,24,.6);
    padding: 7px 14px;
    border-radius: 2px;
    margin-bottom: 10px;
}
.routing-chip .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--gold);
    box-shadow: 0 0 8px var(--gold);
}

/* ---- chat thread / log entries ---- */
.chat-thread { margin-top: 6px; margin-bottom: 20px; }

.log-entry { margin-bottom: 26px; animation: entryIn .45s cubic-bezier(0.22, 1, 0.36, 1); }
@keyframes entryIn { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }

.msg-user {
    margin-left: auto;
    width: fit-content;
    max-width: 78%;
    border: 1px solid rgba(207, 159, 87, 0.20);
    border-right: 2px solid var(--gold);
    background: linear-gradient(160deg, rgba(26, 20, 10, 0.6), rgba(13, 19, 34, 0.5));
    border-radius: 12px;
    padding: 13px 18px;
    margin-bottom: 14px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.35);
}
.msg-user .msg-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 6px;
    text-align: right;
}
.msg-user .msg-text {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    color: var(--ice);
    line-height: 1.6;
}

/* ---- answer panel ---- */
.answer-panel {
    border: 1px solid var(--line);
    border-left: 2px solid var(--gold);
    background: linear-gradient(160deg, rgba(13, 19, 34, 0.72), rgba(7, 10, 20, 0.6));
    border-radius: 12px;
    overflow: hidden;
    padding: 26px 30px;
    position: relative;
    margin-bottom: 6px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.35),
                inset 0 0 24px rgba(207, 159, 87, 0.03);
}
.answer-panel::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, var(--amber-dim), var(--gold-bright), var(--amber-dim));
}
.answer-text {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 16px;
    line-height: 1.8;
    color: var(--text);
    white-space: pre-wrap;
}

/* ---- footer clock ---- */
.mc-footer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--muted-2);
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid var(--line-soft);
}

/* spinner text */
div[data-testid="stSpinner"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: .12em !important;
    color: var(--gold) !important;
    text-transform: uppercase;
}

/* keep the audio element alive but off-screen */
div[data-testid="stAudio"] {
    position: fixed !important;
    bottom: -1000px !important;
    left: -1000px !important;
    height: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* ---- warp / hyperspace HUD (rendered by the canvas warp engine) ---- */
@keyframes warp-pulse { 0%,100% { opacity: .3; } 50% { opacity: .95; } }

.warp-caption {
    position: absolute; bottom: 9%; left: 0; right: 0; text-align: center;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    letter-spacing: .34em; text-transform: uppercase; color: var(--gold);
    text-shadow: 0 0 14px rgba(207,159,87,.45);
    animation: warp-pulse 1.8s ease-in-out infinite;
}
.warp-eta {
    position: absolute; bottom: 5%; left: 0; right: 0; text-align: center;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: .22em; text-transform: uppercase; color: var(--muted);
    opacity: .75;
}
.warp-eta b { color: var(--ice); font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="mc-header">
<div class="mc-brand">
<div class="mc-ring"><div class="mc-core"></div></div>
<div>
<div class="mc-name">ENDURANCE</div>
<div class="mc-name-sub">Interstellar Knowledge Interface</div>
</div>
</div>
<div class="mc-telemetry">
<div>Routing<br><b>AUTO</b></div>
<div>Sources<br><b>&times;3</b></div>
<div class="gold">Relay<br><b>SATURN</b></div>
<div>Link<br><b>STABLE</b></div>
</div>
</div>
<div class="mc-subtitle">Script &nbsp;·&nbsp; The Science of Interstellar &nbsp;·&nbsp; Audience Transmissions</div>
<div class="mc-lede">
Ask anything about the film — plot, characters, the physics behind it, or what
the audience took away from it. A routing agent reads your question and directs
it to the data source best equipped to answer. Ask follow-ups anytime — the
full transmission log stays below.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="hairline">', unsafe_allow_html=True)


# ============================================================
# GRAPH (cached across reruns)
# ============================================================
@st.cache_resource
def get_graph():
    return buildGraph()

graph = get_graph()


# ============================================================
# PERSISTENT CHAT HISTORY — helpers
# ============================================================
def load_sessions():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_sessions(sessions):
    with open(HISTORY_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


# ============================================================
# SESSION STATE — chat history
# ============================================================
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = load_sessions()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "renaming_sid" not in st.session_state:
    st.session_state.renaming_sid = None


def persist_current_session():
    if not st.session_state.messages:
        return
    sid = st.session_state.current_session_id
    existing = st.session_state.all_sessions.get(sid, {})
    # Preserve a user-renamed title; otherwise auto-derive from first question
    if existing.get("title_custom"):
        title = existing.get("title", "Untitled")
    else:
        first_q = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "Untitled")
        title = (first_q[:48] + "…") if len(first_q) > 48 else first_q
    st.session_state.all_sessions[sid] = {
        "title": title,
        "title_custom": existing.get("title_custom", False),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "messages": st.session_state.messages,
    }
    save_sessions(st.session_state.all_sessions)


def rename_session(sid, new_title):
    new_title = (new_title or "").strip()
    if not new_title or sid not in st.session_state.all_sessions:
        return
    st.session_state.all_sessions[sid]["title"] = new_title
    st.session_state.all_sessions[sid]["title_custom"] = True
    save_sessions(st.session_state.all_sessions)


def delete_session(sid):
    st.session_state.all_sessions.pop(sid, None)
    save_sessions(st.session_state.all_sessions)
    # If we deleted the session we're currently viewing, reset to a fresh one
    if st.session_state.current_session_id == sid:
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = []
    if st.session_state.get("renaming_sid") == sid:
        st.session_state.renaming_sid = None


# ============================================================
# SIDEBAR — TRANSMISSION ARCHIVE (chat history navigation)
# ============================================================
with st.sidebar:
    st.markdown('<div class="eyebrow" style="margin-bottom:16px;">ARCHIVE</div>', unsafe_allow_html=True)

    if st.button("➕   New Transmission", key="new_transmission", use_container_width=True):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.renaming_sid = None
        st.rerun()

    st.markdown('<hr class="hairline" style="margin:18px 0;">', unsafe_allow_html=True)

    sorted_sessions = sorted(
        st.session_state.all_sessions.items(),
        key=lambda kv: kv[1].get("updated_at", ""),
        reverse=True,
    )

    if not sorted_sessions:
        st.markdown('<div class="mc-lede" style="font-size:12px;">No past transmissions yet.</div>', unsafe_allow_html=True)

    for sid, sess in sorted_sessions:
        is_active = sid == st.session_state.current_session_id
        title = sess.get("title", "Untitled")

        # Highlight the row for the session currently in view
        if is_active:
            st.markdown(
                f"<style>.st-key-session_{sid} button {{"
                "border-left-color: var(--gold) !important;"
                "color: var(--ice) !important;"
                "background: rgba(10,14,24,.55) !important; }</style>",
                unsafe_allow_html=True,
            )

        if st.session_state.renaming_sid == sid:
            # Inline rename editor
            new_title = st.text_input(
                "Rename transmission",
                value=title,
                key=f"renameinput_{sid}",
                label_visibility="collapsed",
            )
            c_save, c_cancel = st.columns(2)
            if c_save.button("Save", key=f"savename_{sid}", use_container_width=True):
                rename_session(sid, new_title)
                st.session_state.renaming_sid = None
                st.rerun()
            if c_cancel.button("Cancel", key=f"cancelname_{sid}", use_container_width=True):
                st.session_state.renaming_sid = None
                st.rerun()
        else:
            col_sel, col_edit, col_del = st.columns([8, 1, 1])
            label = ("● " if is_active else "") + title
            if col_sel.button(label, key=f"session_{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.session_state.messages = sess.get("messages", [])
                st.rerun()
            if col_edit.button("✎", key=f"rename_{sid}", help="Rename"):
                st.session_state.renaming_sid = sid
                st.rerun()
            if col_del.button("🗑", key=f"del_{sid}", help="Delete"):
                delete_session(sid)
                st.rerun()

        # Timestamp · turn count meta line
        turns = sum(1 for m in sess.get("messages", []) if m.get("role") == "user")
        ts = sess.get("updated_at", "")
        try:
            time_str = datetime.fromisoformat(ts).strftime("%b %d · %H:%M") if ts else ""
        except Exception:
            time_str = ts
        st.markdown(
            f'<div class="archive-meta">{time_str} &nbsp;·&nbsp; {turns} turn{"" if turns == 1 else "s"}</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# WARP OVERLAY + MUSIC HELPERS
# ============================================================
def play_music():
    path = os.path.join(BASE_DIR, "music.mp3")
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.audio(f.read(), format="audio/mp3", autoplay=True, loop=True)


def install_warp_engine():
    """Injects a persistent 3D starfield + hyperspace warp engine into the
    parent page. The canvas drifts at cruise speed behind the UI at all times;
    window.__endurance.engage()/disengage() switch it in and out of warp.
    Runs in the parent document so it survives Streamlit reruns."""
    components.html("""
<script>
(function () {
    const P = window.parent;
    if (P.__endurance) return;

    function engine() {
        if (window.__endurance) return;
        var doc = document;

        /* ---- starfield canvas ---- */
        var cv = doc.createElement('canvas');
        cv.id = 'mc-stars';
        cv.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;';
        doc.body.appendChild(cv);
        var ctx = cv.getContext('2d');
        var DPR = window.devicePixelRatio || 1;
        var W, H, CX, CY;
        function resize() {
            W = cv.width = window.innerWidth * DPR;
            H = cv.height = window.innerHeight * DPR;
            CX = W / 2; CY = H / 2;
        }
        window.addEventListener('resize', resize);
        resize();

        var COUNT = Math.min(700, Math.floor(window.innerWidth * window.innerHeight / 1800));
        var DEPTH = 1600, CRUISE = 1.4, WARP = 62;
        var speed = CRUISE, target = CRUISE;
        var stars = [];
        function spawn(s, rz) {
            s.x = (Math.random() - 0.5) * W * 2.2;
            s.y = (Math.random() - 0.5) * H * 2.2;
            s.z = rz ? Math.random() * DEPTH : DEPTH;
            s.pz = s.z;
            var r = Math.random();
            s.c = r < 0.14 ? 'g' : (r < 0.34 ? 'i' : 'w');
            return s;
        }
        for (var i = 0; i < COUNT; i++) stars.push(spawn({}, true));

        function frame() {
            speed += (target - speed) * (target > speed ? 0.045 : 0.03);
            var wf = Math.min(1, (speed - CRUISE) / (WARP - CRUISE));
            ctx.fillStyle = 'rgba(4,6,12,' + (0.55 - wf * 0.38).toFixed(3) + ')';
            ctx.fillRect(0, 0, W, H);
            for (var k = 0; k < stars.length; k++) {
                var s = stars[k];
                s.pz = s.z;
                s.z -= speed * DPR;
                if (s.z < 1) { spawn(s, false); continue; }
                var sx = CX + (s.x / s.z) * 420, sy = CY + (s.y / s.z) * 420;
                var px = CX + (s.x / s.pz) * 420, py = CY + (s.y / s.pz) * 420;
                if (sx < -50 || sx > W + 50 || sy < -50 || sy > H + 50) { spawn(s, false); continue; }
                var depth = 1 - s.z / DEPTH;
                var a = 0.22 + depth * 0.72;
                var size = (0.4 + depth * 1.8) * DPR;
                var col = s.c === 'g' ? 'rgba(232,184,109,' + a + ')'
                        : s.c === 'i' ? 'rgba(207,224,255,' + a + ')'
                        : 'rgba(232,235,243,' + a + ')';
                if (wf > 0.04) {
                    ctx.strokeStyle = col;
                    ctx.lineWidth = size * (0.7 + wf * 0.9);
                    ctx.lineCap = 'round';
                    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(sx, sy); ctx.stroke();
                } else {
                    ctx.fillStyle = col;
                    ctx.beginPath(); ctx.arc(sx, sy, size * 0.6, 0, Math.PI * 2); ctx.fill();
                }
            }
            window.requestAnimationFrame(frame);
        }
        window.requestAnimationFrame(frame);

        /* ---- gold flash at warp entry/exit ---- */
        var flash = doc.createElement('div');
        flash.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:999995;pointer-events:none;opacity:0;transition:opacity .6s ease;background:radial-gradient(ellipse at center, rgba(240,226,196,.85), rgba(207,159,87,.22) 45%, transparent 75%);';
        doc.body.appendChild(flash);
        function doFlash() {
            flash.style.transition = 'opacity .12s ease';
            flash.style.opacity = '1';
            setTimeout(function () {
                flash.style.transition = 'opacity .7s ease';
                flash.style.opacity = '0';
            }, 170);
        }

        /* ---- warp HUD (dim vignette + caption + elapsed clock) ---- */
        var dim = null, etaIv = null;

        window.__endurance = {
            engage: function (caption) {
                target = WARP;
                doFlash();
                var app = doc.querySelector('.stApp');
                if (app) app.classList.add('mc-warping');
                if (dim) { dim.remove(); dim = null; }
                dim = doc.createElement('div');
                dim.id = 'mc-warp-dim';
                dim.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:999990;opacity:0;transition:opacity .6s ease;background:radial-gradient(ellipse at center, rgba(4,6,12,.15) 25%, rgba(4,6,12,.82) 100%);';
                dim.innerHTML = '<div class="warp-caption">' + (caption || 'Traversing the void &nbsp;&middot;&nbsp; establishing uplink') + '</div>' +
                                '<div class="warp-eta">elapsed <b>00:00</b> &nbsp;&middot;&nbsp; typical 30s&ndash;3min</div>';
                doc.body.appendChild(dim);
                window.requestAnimationFrame(function () { if (dim) dim.style.opacity = '1'; });
                var t0 = Date.now();
                if (etaIv) clearInterval(etaIv);
                etaIv = setInterval(function () {
                    var el = doc.querySelector('#mc-warp-dim .warp-eta');
                    if (!el) { clearInterval(etaIv); etaIv = null; return; }
                    var sec = Math.floor((Date.now() - t0) / 1000);
                    var mm = ('0' + Math.floor(sec / 60)).slice(-2);
                    var ss = ('0' + (sec % 60)).slice(-2);
                    el.innerHTML = 'elapsed <b>' + mm + ':' + ss + '</b> &nbsp;&middot;&nbsp; typical 30s&ndash;3min';
                }, 1000);
            },
            disengage: function (caption) {
                target = CRUISE;
                doFlash();
                if (etaIv) { clearInterval(etaIv); etaIv = null; }
                var app = doc.querySelector('.stApp');
                if (app) app.classList.remove('mc-warping');
                if (dim) {
                    var d = dim; dim = null;
                    var cap = d.querySelector('.warp-caption');
                    if (cap) cap.innerHTML = caption || 'Uplink established';
                    d.style.transition = 'opacity 1.05s ease';
                    d.style.opacity = '0';
                    setTimeout(function () { d.remove(); }, 1200);
                }
            }
        };
    }

    const s = P.document.createElement('script');
    s.textContent = '(' + engine.toString() + ')();';
    P.document.body.appendChild(s);
})();
</script>
""", height=0, width=0)


def warp_command(action, caption=None):
    """Fire an engage/disengage command at the parent-page warp engine.
    Retries briefly in case the engine script is still loading."""
    cap_js = json.dumps(caption) if caption else "null"
    nonce = uuid.uuid4().hex  # force re-execution on every call
    components.html(f"""
<script>
/* {nonce} */
(function () {{
    var tries = 0;
    var iv = setInterval(function () {{
        var E = window.parent.__endurance;
        if (E) {{ clearInterval(iv); E.{action}({cap_js}); }}
        else if (++tries > 120) clearInterval(iv);
    }}, 50);
}})();
</script>
""", height=0, width=0)


def fade_out_audio(duration=1.1):
    components.html(f"""
        <script>
        (function() {{
            try {{
                const doc = window.parent.document;
                const audio = doc.querySelector('div[data-testid="stAudio"] audio');
                if (audio && !audio.paused) {{
                    const steps = 22;
                    const stepTime = ({duration} * 1000) / steps;
                    const startVol = audio.volume || 1;
                    let i = 0;
                    const interval = setInterval(() => {{
                        i += 1;
                        audio.volume = Math.max(0, startVol * (1 - i / steps));
                        if (i >= steps) {{
                            clearInterval(interval);
                            audio.pause();
                        }}
                    }}, stepTime);
                }}
            }} catch (e) {{ /* no-op */ }}
        }})();
        </script>
    """, height=0, width=0)


def run_query(question: str):
    audio_ph = st.empty()
    engage_ph = st.empty()
    fade_ph = st.empty()

    with audio_ph.container():
        play_music()
    with engage_ph.container():
        warp_command("engage")  # canvas stars stretch into hyperspace streaks

    chat_history_payload = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    start = time.time()
    result = graph.invoke({"question": question, "chat_history": chat_history_payload})
    elapsed = time.time() - start

    with fade_ph.container():
        warp_command("disengage", "Uplink established")
        fade_out_audio(1.1)
    time.sleep(1.25)

    engage_ph.empty()
    audio_ph.empty()
    fade_ph.empty()

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("final_answer", "No answer generated."),
        "category": result.get("category", "unknown"),
        "elapsed": elapsed,
    })

    persist_current_session()


CATEGORY_LABELS = {
    "script": "Script Agent — Narrative Archive",
    "textbook": "Textbook Agent — Science Archive",
    "comments": "Comments Agent — Audience Archive",
    "multi": "Multi Agent — Cross-Referenced",
}


# ============================================================
# CHAT INPUT DETECTOR
# ============================================================
user_input = st.chat_input(
    "Transmit your question, e.g. How accurate is the tesseract scene?",
    disabled=st.session_state.is_processing,
)

if user_input and user_input.strip() and not st.session_state.is_processing:
    st.session_state.pending_question = user_input.strip()
    st.session_state.is_processing = True
    st.rerun()

if st.session_state.is_processing and st.session_state.pending_question:
    # FORCE COMPLETE REMOVAL OF INPUT CONTAINER SPACE DURING PROCESSING RUN
    st.markdown(
        '<style>div[data-testid="stBottomBlockContainer"] { display: none !important; }</style>',
        unsafe_allow_html=True,
    )
    run_query(st.session_state.pending_question)
    st.session_state.pending_question = None
    st.session_state.is_processing = False
    st.rerun()


# ============================================================
# TRANSMISSION LOG (chat history)
# ============================================================
if st.session_state.messages:
    st.markdown('<div class="chat-thread">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="log-entry">
                <div class="msg-user">
                    <div class="msg-label">Transmission — Crew Query</div>
                    <div class="msg-text">{html.escape(msg['content'])}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            label = CATEGORY_LABELS.get(msg.get("category", "unknown"), str(msg.get("category", "unknown")).upper())
            elapsed = msg.get("elapsed", 0.0)
            st.markdown(f"""
            <div class="log-entry">
                <div class="routing-chip"><span class="dot"></span>{label} &nbsp;|&nbsp; {elapsed:.1f}s</div>
                <div class="answer-panel">
                    <div class="answer-text">{html.escape(msg['content'])}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="mc-lede">No transmissions yet. Ask a question below to begin.</div>', unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="mc-footer">CSCI 370 &nbsp;·&nbsp; RAG Pipeline Demo &nbsp;·&nbsp; Status: Online &nbsp;·&nbsp; Do not go gentle</div>
""", unsafe_allow_html=True)

# Boot the persistent starfield / warp engine (parent-page canvas)
install_warp_engine()