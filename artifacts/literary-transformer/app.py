import streamlit as st
import streamlit.components.v1 as components
import os
import uuid
from datetime import date
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
)

# ─── Prompts & Maps ────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """당신은 한국 현대 문학의 정수를 담은 작가입니다.
사용자가 일상적인 이야기나 경험을 입력하면, 그것을 실존적인 고독과 서정성이 깃든 현대 소설 문체로 변환해야 합니다.

변환 원칙:
- 한강, 김애란, 편혜영의 문체처럼 내면의 고요한 울림을 담아냅니다
- 일상의 사소한 장면에서 실존적 의미를 길어올립니다
- 감각적이고 구체적인 이미지로 추상적 감정을 표현합니다
- 시제는 주로 현재형을 사용해 독자를 그 순간 안으로 끌어들입니다
- 주인공을 '나'로 지칭하되, 거리감과 관찰자적 시선을 유지합니다
- 자연물이나 일상의 사물에 감정을 투영하는 물아일체적 표현을 활용합니다
- 결말은 열린 결말로, 독자에게 여운을 남깁니다

출력 길이 원칙 (반드시 준수):
- 입력이 짧을수록(한두 문장) 출력도 짧게 씁니다. 한두 문장의 강렬하고 응축된 문학적 이미지로 마무리합니다.
- 입력이 중간 길이이면(서너 문장~한 단락) 출력도 두세 단락 이내로 유지합니다.
- 입력이 길 때만(여러 단락 이상) 긴 호흡의 산문으로 확장합니다.
- 원문 분량의 1.5~2배를 넘지 않는 것을 기본 원칙으로 합니다.

맥락 연속성 원칙:
- 같은 주제 아래 이전 문장들이 있다면, 그 문체·어조·이미지 체계를 완벽하게 이어받아 동일한 화자의 목소리로 씁니다.
- 새로운 일상 이야기가 들어와도 하나의 연속된 산문처럼 자연스럽게 이어지게 합니다.

절대 하지 않을 것:
- 설명하거나 해석하지 않습니다 — 보여줄 뿐입니다
- 감상적 과잉이나 멜로드라마적 표현을 피합니다
- 원본의 일상성을 완전히 버리지 않습니다
- 짧은 입력을 억지로 길게 늘리지 않습니다

반드시 순수한 문학 텍스트만 출력하세요. 설명이나 부연은 없습니다."""

MOOD_MAP = {
    "고독한": "전반적인 분위기는 깊은 고독과 정적입니다. 인물은 세계로부터 미묘하게 분리된 느낌을 줍니다.",
    "따뜻한": "전반적인 분위기는 은은한 온기와 위로입니다. 일상 속 작은 연결과 온도를 섬세하게 담아냅니다.",
    "쓸쓸한": "전반적인 분위기는 잔잔한 슬픔과 그리움입니다. 무언가 이미 지나가 버린 것 같은 여운을 남깁니다.",
    "몽환적인": "전반적인 분위기는 현실과 꿈의 경계가 흐릿한 몽환성입니다. 감각이 뒤섞이고 시간이 느리게 흐릅니다.",
    "긴장된": "전반적인 분위기는 내면의 미세한 긴장과 불안입니다. 일상의 표면 아래 팽팽한 무언가가 숨 쉬고 있습니다.",
}

SEASON_MAP = {
    "봄": "배경의 계절은 봄입니다. 새싹, 꽃가루, 이른 햇살, 아직 차가운 바람 같은 봄의 감각을 자연스럽게 녹여내세요.",
    "여름": "배경의 계절은 여름입니다. 열기, 습도, 짙은 녹음, 소나기 같은 여름의 감각을 자연스럽게 녹여내세요.",
    "가을": "배경의 계절은 가을입니다. 낙엽, 서늘한 바람, 기울어지는 빛, 쓸쓸한 냄새 같은 가을의 감각을 자연스럽게 녹여내세요.",
    "겨울": "배경의 계절은 겨울입니다. 냉기, 눈, 앙상한 나뭇가지, 이른 어둠 같은 겨울의 감각을 자연스럽게 녹여내세요.",
}

TIME_MAP = {
    "새벽": "시간대는 새벽입니다. 세상이 아직 잠든 정적, 가로등 빛, 홀로 깨어 있다는 감각을 활용하세요.",
    "아침": "시간대는 아침입니다. 하루가 막 시작되는 빛과 소리, 아직 정해지지 않은 시간의 질감을 담아내세요.",
    "낮": "시간대는 낮입니다. 뚜렷한 햇빛, 일상의 소음과 분주함, 그 속에서 느끼는 내면의 거리감을 살려내세요.",
    "저녁": "시간대는 저녁입니다. 하루가 기울어지는 빛, 귀가하는 사람들, 하루를 마무리하는 감각을 담아내세요.",
    "밤": "시간대는 밤입니다. 적막, 창문 너머의 불빛, 생각이 깊어지는 밤의 시간을 섬세하게 표현하세요.",
}

MAX_CONTEXT_ENTRIES = 5  # use last N entries as context per session

STORY_COMPLETE_PROMPT = """당신은 한국 현대 문학을 대표하는 단편 작가입니다.
사용자가 여러 개의 독립적인 문학적 장면들을 보내면, 이것들을 하나의 유기적인 단편소설로 엮어야 합니다.

작업 원칙:
- 각 장면의 고유한 문체·어조·이미지를 최대한 살리면서 자연스럽게 연결합니다
- 장면과 장면 사이에 필요한 경우에만 최소한의 연결 문장을 삽입합니다
- 억지스러운 플롯을 만들지 않습니다 — 흐름이 있되, 설명하지 않습니다
- 전체가 동일한 화자의 목소리로 읽혀야 합니다
- 결말은 열린 결말로 두어 여운을 남깁니다
- 단편의 제목을 맨 앞에 붙이되, 제목 뒤에 빈 줄 하나를 두세요 (형식: [제목]\n\n[본문])
- 제목은 주어진 주제명을 그대로 쓰거나, 더 문학적인 제목으로 새로 짓되 짧고 함축적이어야 합니다

절대 하지 않을 것:
- 각 장면에 번호나 소제목을 붙이지 않습니다
- 원문의 감각적 이미지를 설명으로 대체하지 않습니다
- 과도한 서사적 연결로 작품의 여백을 채우지 않습니다

반드시 순수한 문학 텍스트만 출력하세요. 설명이나 부연은 없습니다."""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_session(title: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "entries": [],        # list of {"input": str, "output": str}
        "completed_story": None,  # str | None — the woven short story
        "mood": None,
        "season": None,
        "time_of_day": None,
    }


def build_system_prompt(mood, season, time_of_day) -> str:
    hints = []
    if mood and mood in MOOD_MAP:
        hints.append(MOOD_MAP[mood])
    if season and season in SEASON_MAP:
        hints.append(SEASON_MAP[season])
    if time_of_day and time_of_day in TIME_MAP:
        hints.append(TIME_MAP[time_of_day])
    if hints:
        return BASE_SYSTEM_PROMPT + "\n\n[분위기 힌트]\n" + "\n".join(f"- {h}" for h in hints)
    return BASE_SYSTEM_PROMPT


def get_length_tokens(n: int) -> int:
    if n <= 80:   return 300
    if n <= 200:  return 600
    if n <= 500:  return 1200
    if n <= 1000: return 2400
    return 8192


def get_length_hint(n: int) -> str:
    if n <= 80:   return "매우 짧은 입력입니다. 한두 문장의 응축된 이미지로만 표현하세요. 절대 길게 늘리지 마세요."
    if n <= 200:  return "짧은 입력입니다. 세 문장 이내로 간결하게 표현하세요."
    if n <= 500:  return "중간 길이 입력입니다. 두세 단락 이내로 표현하세요."
    return "긴 입력입니다. 긴 호흡의 산문으로 풍부하게 표현해도 좋습니다."


def get_full_text(session: dict) -> str:
    return "\n\n".join(e["output"] for e in session["entries"])


def session_chat_history(session: dict, exclude_last: bool = False) -> list[dict]:
    entries = session["entries"]
    if exclude_last and entries:
        entries = entries[:-1]
    # limit to last MAX_CONTEXT_ENTRIES
    entries = entries[-MAX_CONTEXT_ENTRIES:]
    msgs = []
    for e in entries:
        msgs.append({"role": "user", "content": e["input"]})
        msgs.append({"role": "assistant", "content": e["output"]})
    return msgs


def call_api(user_text: str, session: dict, is_retry: bool = False) -> str:
    n = len(user_text.strip())
    system_prompt = build_system_prompt(session.get("mood"), session.get("season"), session.get("time_of_day"))
    length_hint = get_length_hint(n)
    max_tokens = get_length_tokens(n)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    history = session_chat_history(session, exclude_last=is_retry)
    messages.extend(history)

    if is_retry:
        content = (
            f"[길이 지침: {length_hint}]\n\n"
            "[재시도: 앞의 문체·어조를 그대로 유지하면서 같은 원문을 다른 문학적 표현으로 다시 써주세요.]\n\n"
            f"{user_text.strip()}"
        )
    else:
        content = f"[길이 지침: {length_hint}]\n\n{user_text.strip()}"

    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model="gpt-5.1",
        max_completion_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def make_share_card(story_title: str, story_body: str, pen_name: str = "") -> str:
    today = date.today().strftime("%Y.%m.%d")
    border = "─" * 28
    byline = f"  {pen_name}  ·  {today}" if pen_name.strip() else f"  {today}"
    return (
        f"{border}\n"
        f"  {story_title}\n"
        f"{border}\n\n"
        f"{story_body}\n\n"
        f"{border}\n"
        f"{byline}\n"
        f"  일상의 문학\n"
        f"{border}"
    )


def call_story_complete_api(session: dict) -> str:
    entries = session["entries"]
    scenes = "\n\n---\n\n".join(e["output"] for e in entries)
    hint_parts = [h for h in [session.get("mood"), session.get("season"), session.get("time_of_day")] if h]
    hint_line = ("분위기 힌트: " + " · ".join(hint_parts) + "\n\n") if hint_parts else ""

    user_content = (
        f"주제명: {session['title']}\n\n"
        f"{hint_line}"
        f"아래는 이 주제 아래 쌓인 {len(entries)}개의 문학적 장면들입니다. "
        f"이것들을 하나의 완성된 단편소설로 엮어주세요.\n\n"
        f"{scenes}"
    )

    response = client.chat.completions.create(
        model="gpt-5.1",
        max_completion_tokens=8192,
        messages=[
            {"role": "system", "content": STORY_COMPLETE_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="일상의 문학", page_icon="✦", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Noto Serif KR', Georgia, serif; }
    .stApp { background-color: #FAF8F5; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #F0EDE8 !important;
        border-right: 1px solid #D4C8BE;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        font-family: 'Noto Serif KR', Georgia, serif;
    }

    .sidebar-title {
        font-size: 1.1rem; font-weight: 500; color: #2C2C2C;
        letter-spacing: 0.1em; margin-bottom: 0.2rem;
    }
    .sidebar-sub {
        font-size: 0.72rem; color: #9A8880; letter-spacing: 0.2em;
        margin-bottom: 1.5rem;
    }
    .session-item {
        padding: 0.55rem 0.8rem;
        border-radius: 2px;
        cursor: pointer;
        margin-bottom: 0.3rem;
        border-left: 2px solid transparent;
    }
    .session-item.active {
        background: #FAF8F5;
        border-left: 2px solid #8B6F5E;
    }
    .session-item-title { font-size: 0.88rem; color: #2C2C2C; font-weight: 400; }
    .session-item-meta  { font-size: 0.7rem;  color: #9A8880; margin-top: 0.1rem; }

    /* ── Main area ── */
    .main-inner { max-width: 760px; margin: 0 auto; padding: 2rem 1rem 4rem 1rem; }

    .session-header {
        border-bottom: 1px solid #D4C8BE;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    .session-title-text {
        font-size: 1.6rem; font-weight: 300; color: #2C2C2C;
        letter-spacing: 0.12em; margin: 0;
    }
    .session-meta-row {
        font-size: 0.72rem; color: #9A8880; letter-spacing: 0.2em;
        margin-top: 0.4rem;
    }

    /* ── Entry display ── */
    .entry-block {
        margin-bottom: 2rem;
    }
    .entry-output {
        line-height: 2.1; font-size: 1.02rem; color: #2C2C2C;
        font-weight: 300; letter-spacing: 0.02em; white-space: pre-wrap;
    }
    .entry-source {
        font-size: 0.74rem; color: #B0A49C; letter-spacing: 0.05em;
        margin-top: 0.5rem; font-style: italic;
        border-left: 2px solid #E0D8D2; padding-left: 0.7rem;
    }
    .entry-divider {
        border: none; border-top: 1px dashed #E0D8D2; margin: 1.5rem 0;
    }

    .section-label {
        font-size: 0.72rem; color: #9A8880; letter-spacing: 0.3em;
        text-transform: uppercase; margin-bottom: 0.6rem; font-weight: 400;
    }

    /* ── Hint box ── */
    .hint-section {
        background-color: #F5F2EE; border: 1px solid #E0D8D2;
        border-radius: 2px; padding: 1rem 1.2rem 0.6rem 1.2rem; margin-bottom: 1rem;
    }
    .hint-title {
        font-size: 0.7rem; color: #9A8880; letter-spacing: 0.25em;
        text-transform: uppercase; margin-bottom: 0.7rem; font-weight: 400;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'Noto Serif KR', Georgia, serif !important;
        background-color: #2C2C2C !important; color: #FAF8F5 !important;
        border: none !important; border-radius: 2px !important;
        font-size: 0.8rem !important; letter-spacing: 0.2em !important;
        font-weight: 400 !important; padding: 0.6rem 1.5rem !important;
        transition: background-color 0.2s ease !important; width: 100%;
    }
    .stButton > button:hover  { background-color: #8B6F5E !important; }
    .stButton > button:active { background-color: #6B4F3E !important; }

    .ghost-btn button {
        background-color: transparent !important; color: #9A8880 !important;
        border: 1px solid #D4C8BE !important; font-size: 0.75rem !important;
        padding: 0.4rem 1rem !important; letter-spacing: 0.15em !important;
    }
    .ghost-btn button:hover { background-color: #EDE8E2 !important; color: #6B4F3E !important; }

    .stTextArea textarea {
        font-family: 'Noto Serif KR', Georgia, serif !important;
        font-size: 0.95rem !important; line-height: 1.85 !important;
        color: #2C2C2C !important; background-color: #FFFFFF !important;
        border: 1px solid #D4C8BE !important; border-radius: 2px !important;
        padding: 1rem 1.2rem !important; caret-color: #8B6F5E;
    }
    .stTextArea textarea:focus {
        border-color: #8B6F5E !important;
        box-shadow: 0 0 0 1px #8B6F5E22 !important;
    }
    .stTextInput input {
        font-family: 'Noto Serif KR', Georgia, serif !important;
        font-size: 0.95rem !important; color: #2C2C2C !important;
        background-color: #FFFFFF !important; border: 1px solid #D4C8BE !important;
        border-radius: 2px !important; padding: 0.6rem 1rem !important;
        caret-color: #8B6F5E;
    }
    .stTextInput input:focus {
        border-color: #8B6F5E !important;
        box-shadow: 0 0 0 1px #8B6F5E22 !important;
    }

    div[data-testid="stSpinner"] { text-align: center; }
    .spinner-text {
        font-size: 0.8rem; color: #9A8880; letter-spacing: 0.2em;
        text-align: center; margin: 1rem 0;
        animation: fadeInOut 2s ease-in-out infinite;
    }
    @keyframes fadeInOut { 0%,100%{opacity:.4} 50%{opacity:1} }

    div[data-testid="stSegmentedControl"] { font-family: 'Noto Serif KR', Georgia, serif !important; }
    div[data-testid="stSegmentedControl"] label { font-size: 0.78rem !important; }

    .full-text-box {
        background: #FFFFFF; border: 1px solid #E0D8D2; border-radius: 2px;
        padding: 2rem 2.2rem; line-height: 2.1; font-size: 1rem;
        color: #2C2C2C; font-weight: 300; white-space: pre-wrap;
        max-height: 500px; overflow-y: auto;
    }

    .divider { border: none; border-top: 1px solid #E0D8D2; margin: 2rem 0; }
    [data-testid="stMarkdownContainer"] p { font-family: 'Noto Serif KR', Georgia, serif; }

    /* ── Completed story view ── */
    .story-box {
        background: #FFFFFF;
        border: 1px solid #D4C8BE;
        border-radius: 2px;
        padding: 2.5rem 2.8rem;
        line-height: 2.3;
        font-size: 1.05rem;
        color: #2C2C2C;
        font-weight: 300;
        letter-spacing: 0.025em;
        white-space: pre-wrap;
    }
    .story-title-text {
        font-size: 1.3rem;
        font-weight: 400;
        color: #2C2C2C;
        letter-spacing: 0.15em;
        margin: 0 0 0.3rem 0;
        text-align: center;
    }
    .story-byline {
        text-align: center;
        font-size: 0.78rem;
        color: #8B6F5E;
        letter-spacing: 0.3em;
        margin: 0.25rem 0 0 0;
        font-weight: 400;
    }
    .story-ornament {
        text-align: center;
        color: #B0A49C;
        font-size: 0.85rem;
        letter-spacing: 0.5em;
        margin-bottom: 2rem;
    }
    .story-body {
        line-height: 2.3;
        font-size: 1.02rem;
        color: #2C2C2C;
        font-weight: 300;
        letter-spacing: 0.02em;
        white-space: pre-wrap;
    }
    .complete-cta {
        background: linear-gradient(135deg, #F5F2EE 0%, #EDE8E2 100%);
        border: 1px solid #D4C8BE;
        border-radius: 2px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .complete-cta-title {
        font-size: 1rem; font-weight: 400; color: #2C2C2C;
        letter-spacing: 0.1em; margin-bottom: 0.5rem;
    }
    .complete-cta-sub {
        font-size: 0.8rem; color: #9A8880; letter-spacing: 0.05em;
        line-height: 1.8; margin-bottom: 1.2rem;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #D4C8BE;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Noto Serif KR', Georgia, serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.2em !important;
        color: #9A8880 !important;
        background: transparent !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2C2C2C !important;
        border-bottom: 2px solid #8B6F5E !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

    /* ── Share section ── */
    .share-section {
        margin-top: 2rem;
        border-top: 1px dashed #D4C8BE;
        padding-top: 1.5rem;
    }
    .share-label {
        font-size: 0.72rem; color: #9A8880; letter-spacing: 0.3em;
        text-transform: uppercase; margin-bottom: 1rem; font-weight: 400;
    }
    .share-card-preview {
        background: #FAFAF8;
        border: 1px solid #E0D8D2;
        border-radius: 2px;
        padding: 1.4rem 1.6rem;
        font-family: 'Noto Serif KR', Georgia, serif;
        font-size: 0.88rem;
        line-height: 1.95;
        color: #4A4440;
        white-space: pre-wrap;
        letter-spacing: 0.02em;
        margin-bottom: 1rem;
    }
    .copy-btn-wrap button {
        background: transparent !important;
        color: #8B6F5E !important;
        border: 1px solid #C4B8B0 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.2em !important;
        padding: 0.45rem 1.2rem !important;
        border-radius: 2px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-family: 'Noto Serif KR', Georgia, serif !important;
    }
    .copy-btn-wrap button:hover {
        background: #EDE8E2 !important;
        border-color: #8B6F5E !important;
    }
    /* Streamlit expander styling */
    .stExpander {
        border: 1px solid #E0D8D2 !important;
        border-radius: 2px !important;
        background: #FAF8F5 !important;
    }
    .stExpander summary {
        font-family: 'Noto Serif KR', Georgia, serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.2em !important;
        color: #9A8880 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ────────────────────────────────────────────────────────

if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "active_idx" not in st.session_state:
    st.session_state.active_idx = None
if "creating_session" not in st.session_state:
    st.session_state.creating_session = False
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "pen_name" not in st.session_state:
    st.session_state.pen_name = ""


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-title">일상의 문학</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">나의 문학 노트</p>', unsafe_allow_html=True)

    if st.button("＋  새 주제 만들기", use_container_width=True, key="new_topic_btn"):
        st.session_state.creating_session = True
        st.session_state.last_result = ""

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pen name ──
    st.markdown('<p class="section-label">작가 이름 / 펜네임</p>', unsafe_allow_html=True)
    pen_input = st.text_input(
        label="펜네임",
        value=st.session_state.pen_name,
        placeholder="예: 김도윤, 밤의 작가, anonymous…",
        label_visibility="collapsed",
        key="pen_name_input",
    )
    if pen_input != st.session_state.pen_name:
        st.session_state.pen_name = pen_input
    if st.session_state.pen_name.strip():
        st.markdown(
            f'<p style="font-size:0.72rem; color:#8B6F5E; letter-spacing:0.05em; margin-top:0.2rem;">'
            f'✦ {st.session_state.pen_name} 으로 저장됩니다</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.sessions:
        st.markdown('<p class="section-label">저장된 주제</p>', unsafe_allow_html=True)
        for i, sess in enumerate(reversed(st.session_state.sessions)):
            real_idx = len(st.session_state.sessions) - 1 - i
            is_active = real_idx == st.session_state.active_idx
            entry_count = len(sess["entries"])
            count_label = f"{entry_count}개의 문장" if entry_count else "아직 비어있음"
            bg = "background:#FAF8F5; border-left:2px solid #8B6F5E;" if is_active else "border-left:2px solid transparent;"
            st.markdown(
                f'<div class="session-item" style="{bg} padding:0.55rem 0.8rem; margin-bottom:0.3rem; border-radius:2px;">'
                f'<div class="session-item-title">{sess["title"]}</div>'
                f'<div class="session-item-meta">{count_label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("열기", key=f"open_{sess['id']}", use_container_width=True):
                st.session_state.active_idx = real_idx
                st.session_state.creating_session = False
                st.session_state.last_result = ""
                st.rerun()
    else:
        st.markdown(
            '<p style="font-size:0.78rem; color:#B0A49C; letter-spacing:0.05em;">아직 주제가 없습니다.<br>새 주제를 만들어 보세요.</p>',
            unsafe_allow_html=True,
        )


# ─── Main content ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-inner">', unsafe_allow_html=True)

# ── CREATE NEW SESSION ─────────────────────────────────────────────────────────
if st.session_state.creating_session:
    st.markdown('<div class="session-header"><p class="session-title-text">새 주제 만들기</p></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">주제 제목</p>', unsafe_allow_html=True)

    new_title = st.text_input(
        label="주제 제목",
        placeholder="예: 출퇴근길의 기억, 혼자였던 오후, 어느 겨울의 단편...",
        label_visibility="collapsed",
        key="new_title_input",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="hint-section">', unsafe_allow_html=True)
    st.markdown('<p class="hint-title">✦ 이 주제의 분위기 힌트 (선택)</p>', unsafe_allow_html=True)

    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        new_mood = st.segmented_control("분위기", list(MOOD_MAP.keys()), default=None, key="new_mood")
    with hc2:
        new_season = st.segmented_control("계절", list(SEASON_MAP.keys()), default=None, key="new_season")
    with hc3:
        new_time = st.segmented_control("시간대", list(TIME_MAP.keys()), default=None, key="new_time")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    bc1, bc2 = st.columns([3, 1])
    with bc1:
        if st.button("✦  이 주제로 시작하기", use_container_width=True, key="confirm_new"):
            title = new_title.strip() if new_title and new_title.strip() else "제목 없는 주제"
            sess = make_session(title)
            sess["mood"] = new_mood
            sess["season"] = new_season
            sess["time_of_day"] = new_time
            st.session_state.sessions.append(sess)
            st.session_state.active_idx = len(st.session_state.sessions) - 1
            st.session_state.creating_session = False
            st.session_state.last_result = ""
            st.rerun()
    with bc2:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("취소", use_container_width=True, key="cancel_new"):
            st.session_state.creating_session = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ── ACTIVE SESSION ─────────────────────────────────────────────────────────────
elif st.session_state.active_idx is not None and st.session_state.sessions:
    idx = st.session_state.active_idx
    if idx >= len(st.session_state.sessions):
        st.session_state.active_idx = len(st.session_state.sessions) - 1
        idx = st.session_state.active_idx

    session = st.session_state.sessions[idx]
    # Migrate old sessions that don't have completed_story
    if "completed_story" not in session:
        session["completed_story"] = None

    entries = session["entries"]
    entry_count = len(entries)

    active_hints = " · ".join(
        h for h in [session.get("mood"), session.get("season"), session.get("time_of_day")] if h
    )
    meta = f"{entry_count}개의 문장"
    if active_hints:
        meta += f" &nbsp;·&nbsp; {active_hints}"
    if session.get("completed_story"):
        meta += " &nbsp;·&nbsp; ✦ 단편 완성됨"

    st.markdown(
        f'<div class="session-header">'
        f'<p class="session-title-text">{session["title"]}</p>'
        f'<p class="session-meta-row">{meta}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_manuscript, tab_story = st.tabs(["원고", "단편 완성본"])

    # ══ TAB 1: 원고 (Manuscript) ══════════════════════════════════════════════
    with tab_manuscript:
        if entries:
            for i, entry in enumerate(entries):
                st.markdown(
                    f'<div class="entry-block">'
                    f'<div class="entry-output">{entry["output"]}</div>'
                    f'<div class="entry-source">원문: {entry["input"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if i < len(entries) - 1:
                    st.markdown('<hr class="entry-divider">', unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:#B0A49C; font-size:0.88rem; letter-spacing:0.05em; margin:2rem 0;">'
                '아직 문장이 없습니다. 아래에서 첫 번째 이야기를 써보세요.</p>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">새 이야기 추가</p>', unsafe_allow_html=True)

        user_input = st.text_area(
            label="새 이야기",
            placeholder="오늘 있었던 일을 자유롭게 적어주세요...",
            height=120,
            key=f"input_{session['id']}",
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✦  문학으로 변환하여 추가", use_container_width=True, key="transform_btn"):
            if not user_input or not user_input.strip():
                st.warning("이야기를 먼저 입력해 주세요.")
            else:
                with st.spinner(""):
                    st.markdown('<p class="spinner-text">문장을 다듬는 중입니다…</p>', unsafe_allow_html=True)
                    try:
                        result = call_api(user_input, session, is_retry=False)
                        session["entries"].append({"input": user_input.strip(), "output": result})
                        session["completed_story"] = None  # invalidate on new entry
                        st.session_state.last_result = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {str(e)}")

        if entries:
            st.markdown("<br>", unsafe_allow_html=True)
            ac1, ac2, ac3, ac4 = st.columns([2, 2, 2, 1])

            with ac1:
                if st.button("마지막 문장 다시 변환", key="retry_last", use_container_width=True):
                    last_input = entries[-1]["input"]
                    session["entries"] = entries[:-1]
                    with st.spinner(""):
                        try:
                            result = call_api(last_input, session, is_retry=True)
                            session["entries"].append({"input": last_input, "output": result})
                            session["completed_story"] = None
                            st.session_state.last_result = result
                            st.rerun()
                        except Exception as e:
                            session["entries"].append({"input": last_input, "output": entries[-1]["output"]})
                            st.error(f"오류가 발생했습니다: {str(e)}")

            with ac2:
                raw_text = f"[ {session['title']} ]\n\n" + get_full_text(session)
                st.download_button(
                    label="원고 저장 (.txt)",
                    data=raw_text,
                    file_name=f"{session['title']}_원고.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_raw_btn",
                )

            with ac3:
                st.text_area(
                    label="복사용",
                    value=get_full_text(session),
                    height=80,
                    key="copy_area_raw",
                    help="클릭 후 전체 선택(Ctrl+A)으로 복사하세요",
                    label_visibility="collapsed",
                )

            with ac4:
                st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
                if st.button("마지막\n삭제", key="delete_last", use_container_width=True):
                    if entries:
                        session["entries"] = entries[:-1]
                        session["completed_story"] = None
                        st.session_state.last_result = ""
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # ══ TAB 2: 단편 완성본 ═════════════════════════════════════════════════════
    with tab_story:
        completed = session.get("completed_story")

        if not entries:
            st.markdown(
                '<p style="color:#B0A49C; font-size:0.88rem; letter-spacing:0.05em; margin:2rem 0;">'
                '"원고" 탭에서 문장을 먼저 쌓아주세요. 2개 이상의 문장이 있어야 단편을 완성할 수 있습니다.</p>',
                unsafe_allow_html=True,
            )
        elif len(entries) < 2 and not completed:
            st.markdown(
                '<p style="color:#B0A49C; font-size:0.88rem; letter-spacing:0.05em; margin:2rem 0;">'
                '문장이 최소 2개 이상 있어야 단편을 완성할 수 있습니다.</p>',
                unsafe_allow_html=True,
            )
        else:
            if completed:
                # ── Split title and body ──
                lines = completed.split("\n", 1)
                if len(lines) == 2:
                    story_title = lines[0].strip().strip("[]「」『』《》〈〉")
                    story_body = lines[1].strip()
                else:
                    story_title = session["title"]
                    story_body = completed.strip()

                # ── Story reading view ──
                pen = st.session_state.pen_name.strip()
                byline_html = (
                    f'<p class="story-byline">{pen}</p>' if pen else ""
                )
                st.markdown(
                    f'<div class="story-box">'
                    f'<p class="story-title-text">{story_title}</p>'
                    f'{byline_html}'
                    f'<p class="story-ornament">✦ &nbsp; ✦ &nbsp; ✦</p>'
                    f'<div class="story-body">{story_body}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Primary actions ──
                ex1, ex2 = st.columns([3, 2])
                with ex1:
                    st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
                    if st.button("✦  단편 다시 완성하기", key="regen_story", use_container_width=True):
                        with st.spinner(""):
                            st.markdown('<p class="spinner-text">단편을 다시 엮는 중입니다…</p>', unsafe_allow_html=True)
                            try:
                                session["completed_story"] = call_story_complete_api(session)
                                st.rerun()
                            except Exception as e:
                                st.error(f"오류가 발생했습니다: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with ex2:
                    pen_dl = st.session_state.pen_name.strip()
                    dl_header = f"[ {story_title} ]"
                    if pen_dl:
                        dl_header += f"\n  {pen_dl}"
                    dl_content = f"{dl_header}\n\n{story_body}"
                    st.download_button(
                        label="단편 저장 (.txt)",
                        data=dl_content,
                        file_name=f"{story_title}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="download_story_btn",
                    )

                # ── Share section ──
                st.markdown('<div class="share-section">', unsafe_allow_html=True)
                st.markdown('<p class="share-label">공유하기</p>', unsafe_allow_html=True)

                share_card = make_share_card(story_title, story_body, st.session_state.pen_name)

                # Clipboard copy button via JS
                escaped = share_card.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
                components.html(
                    f"""
                    <style>
                      @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400&display=swap');
                      body {{ margin: 0; padding: 0; background: transparent; }}
                      .wrap {{
                        display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
                      }}
                      button {{
                        font-family: 'Noto Serif KR', Georgia, serif;
                        background: transparent;
                        color: #8B6F5E;
                        border: 1px solid #C4B8B0;
                        font-size: 0.76rem;
                        letter-spacing: 0.18em;
                        padding: 7px 18px;
                        border-radius: 2px;
                        cursor: pointer;
                        transition: all 0.2s ease;
                      }}
                      button:hover {{ background: #EDE8E2; border-color: #8B6F5E; }}
                      button.copied {{ color: #5E7A6B; border-color: #5E7A6B; }}
                      .hint {{
                        font-family: 'Noto Serif KR', Georgia, serif;
                        font-size: 0.72rem;
                        color: #B0A49C;
                        letter-spacing: 0.05em;
                      }}
                    </style>
                    <div class="wrap">
                      <button id="copyBtn" onclick="copyText()">클립보드에 복사</button>
                      <span class="hint">카카오톡 · 메모 · 이메일 등 어디든 붙여넣기 가능합니다</span>
                    </div>
                    <script>
                      const text = `{escaped}`;
                      function copyText() {{
                        navigator.clipboard.writeText(text).then(() => {{
                          const btn = document.getElementById('copyBtn');
                          btn.textContent = '✦ 복사됨';
                          btn.classList.add('copied');
                          setTimeout(() => {{
                            btn.textContent = '클립보드에 복사';
                            btn.classList.remove('copied');
                          }}, 2200);
                        }}).catch(() => {{
                          const ta = document.createElement('textarea');
                          ta.value = text;
                          document.body.appendChild(ta);
                          ta.select();
                          document.execCommand('copy');
                          document.body.removeChild(ta);
                          const btn = document.getElementById('copyBtn');
                          btn.textContent = '✦ 복사됨';
                          btn.classList.add('copied');
                          setTimeout(() => {{
                            btn.textContent = '클립보드에 복사';
                            btn.classList.remove('copied');
                          }}, 2200);
                        }});
                      }}
                    </script>
                    """,
                    height=52,
                )

                # Preview of the share card
                with st.expander("공유용 카드 미리보기"):
                    st.markdown(
                        f'<div class="share-card-preview">{share_card}</div>',
                        unsafe_allow_html=True,
                    )
                    st.text_area(
                        label="텍스트 직접 복사",
                        value=share_card,
                        height=200,
                        key="share_card_area",
                        help="전체 선택(Ctrl+A) 후 복사하세요",
                        label_visibility="collapsed",
                    )

                st.markdown('</div>', unsafe_allow_html=True)

            else:
                # ── Call-to-action to generate ──
                st.markdown(
                    f'<div class="complete-cta">'
                    f'<p class="complete-cta-title">단편 완성하기</p>'
                    f'<p class="complete-cta-sub">'
                    f'지금까지 쌓인 {len(entries)}개의 문장을<br>'
                    f'하나의 흐르는 단편소설로 엮어드립니다.<br>'
                    f'장면 사이의 빈 공간을 메우고, 하나의 목소리로 완성합니다.'
                    f'</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("✦  단편 완성하기", use_container_width=True, key="complete_story_btn"):
                    with st.spinner(""):
                        st.markdown('<p class="spinner-text">단편을 엮는 중입니다… 잠시만 기다려 주세요.</p>', unsafe_allow_html=True)
                        try:
                            session["completed_story"] = call_story_complete_api(session)
                            st.rerun()
                        except Exception as e:
                            st.error(f"오류가 발생했습니다: {str(e)}")

# ── EMPTY STATE ────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; padding: 5rem 2rem;">
        <p style="font-size:1.8rem; font-weight:300; color:#2C2C2C; letter-spacing:0.15em; margin-bottom:0.5rem;">일상의 문학</p>
        <p style="font-size:0.82rem; color:#9A8880; letter-spacing:0.2em; margin-bottom:3rem;">평범한 하루를 문학으로 · 일상을 서사로</p>
        <p style="font-size:0.9rem; color:#B0A49C; letter-spacing:0.05em; line-height:2;">
            왼쪽 상단의 <strong>＋ 새 주제 만들기</strong>를 눌러<br>
            오늘의 이야기를 시작해 보세요.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
