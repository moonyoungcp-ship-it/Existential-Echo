import streamlit as st
import streamlit.components.v1 as components
import os
import uuid
import json
import shutil
from pathlib import Path
from datetime import date, datetime
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _get_client():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        key = st.session_state.get("_api_key_input", "")
    if not key:
        return None
    return OpenAI(api_key=key)

# ─── Persistence ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
DATA_FILE   = DATA_DIR / "userdata.json"
MAX_BACKUPS = 3

def _backup_path(n: int) -> Path:
    return DATA_DIR / f"userdata.backup.{n}.json"

def save_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for n in range(MAX_BACKUPS, 1, -1):
        src = _backup_path(n - 1)
        if src.exists():
            shutil.copy2(src, _backup_path(n))
    if DATA_FILE.exists():
        shutil.copy2(DATA_FILE, _backup_path(1))
    
    payload = {
        "sessions":   st.session_state.get("sessions", []),
        "active_idx": st.session_state.get("active_idx"),
        "pen_name":   st.session_state.get("pen_name", ""),
        "saved_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    DATA_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state["_last_saved_at"] = payload["saved_at"]

def load_data(from_backup: int = 0) -> tuple[dict, str]:
    candidates: list[tuple[Path, str]] = [(DATA_FILE, "main")]
    for n in range(1, MAX_BACKUPS + 1):
        candidates.append((_backup_path(n), f"backup.{n}"))
    if from_backup > 0:
        candidates = [(_backup_path(from_backup), f"backup.{from_backup}")]
    for path, label in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data.get("sessions"), list):
                    return data, label
            except Exception:
                continue
    return {"sessions": [], "active_idx": None, "pen_name": "", "saved_at": ""}, "empty"

# ─── Prompts & Maps ────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """당신은 한국 현대 문학의 정수를 담은 작가입니다.
사용자가 일상적인 이야기나 경험을 입력하면, 그것을 실존적인 고독과 서정성이 깃든 현대 소설 문체로 변환해야 합니다.
설명하거나 해석하지 않습니다 — 보여줄 뿐입니다. 반드시 순수한 문학 텍스트만 출력하세요. 별표(**) 같은 기호는 금지합니다."""

AUTO_ENGINE_PROMPT = """당신은 소설 창작의 전 과정을 완벽하게 통제하는 수석 문학 감독이자 거장 소설가입니다.
작가가 제공한 시놉시스를 바탕으로, 지정된 분량 호흡에 맞추어 [1단계: 인물 구축], [2단계: 세부 배경 묘사], [3단계: 갈등 및 사건 전개], [4단계: 최종 문장화 및 합성] 단계를 정밀하게 수행해야 합니다.

소설 분량별 서사 페이스 조절 규칙:
1. 단편 소설: 중심 갈등의 뼈대를 밀도 높게 압축하여 빠르게 빌드업합니다.
2. 중편 소설: 갈등이 고조되는 정황과 심리 변화를 서두르지 않고 서너 단락에 걸쳐 차분하게 전개합니다.
3. 장편 소설: 서사를 절대 서둘러 결론짓지 마십시오. 인물이 처한 방의 온도, 가구의 냄새, 피부에 닿는 계절감, 인물의 아주 사소한 손짓과 깊은 전사(過去)까지 극도로 세밀하고 장엄하게 빌드업하십시오.

절대 주의 사항:
출력하는 결과물 텍스트 전체에 별표(**) 같은 마크다운 강조 기호를 절대로 섞지 마십시오. 오직 정갈한 순수 문장만 출력해야 합니다."""

STORY_COMPLETE_PROMPT = """당신은 한국 현대 문학을 대표하는 단편 작가입니다.
사용자가 여러 개의 독립적인 문학적 장면들을 보내면, 이것들을 하나의 유기적인 단편소설로 엮어야 합니다.
반드시 문장 앞뒤나 내부에 별표(**) 같은 마크다운 표식을 절대 사용하지 마세요. 순수한 문학 텍스트만 출력하세요."""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_session(title: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "entries": [],
        "completed_story": None,
        "mood": None,
        "season": None,
        "time_of_day": None,
        "style_mode": None,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "synopsis": "",
        "auto_steps": {"1": "", "2": "", "3": "", "4": ""},
        "auto_length": "단편 소설",
        "auto_style": "성애나"
    }

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def get_full_text(session: dict) -> str:
    return "\n\n".join(e["output"] for e in session["entries"])

def call_api(user_text: str, session: dict) -> str:
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    response = client.chat.completions.create(
        model="gpt-4o", max_completion_tokens=2000,
        messages=[{"role": "system", "content": BASE_SYSTEM_PROMPT}, {"role": "user", "content": user_text.strip()}]
    )
    return response.choices[0].message.content.strip().replace("**", "")

def call_auto_engine_api(session: dict, step: str) -> str:
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    
    step_instruction = {
        "1": "1단계 [인물 구축]: 제공된 소설 시놉시스를 바탕으로 주인공 및 주변 인물들의 성격, 심리적 결함, 모순적 내면을 깊이 있게 분석하고 설정안을 도출해라.",
        "2": f"2단계 [세부 배경 묘사]: 앞서 구축된 인물 정보({session['auto_steps']['1']})를 참조하여, 그들이 호흡할 공간의 대기, 온도, 습도, 사물의 감각적 풍경을 세밀하게 빌드업해라.",
        "3": f"3단계 [갈등 및 사건 전개]: 앞선 인물과 배경 설정({session['auto_steps']['2']}) 위에서 시놉시스의 사건이 어떻게 고조되는지 구체적인 서사 타임라인 및 심리적 균열을 설계해라.",
        "4": f"4단계 [최종 문장화 및 합성]: 모든 빌드업 데이터({session['auto_steps']['3']})를 종합하여, 선택된 '{session.get('auto_style', '성애나')}'의 문체 톤으로 실제 소설 원고 본문 문단(4~5문장 이상)을 완벽하게 가공하여 완성해라."
    }
    
    user_content = (
        f"소설 기획 제목: {session['title']}\n"
        f"설정된 소설 분량 규격: {session.get('auto_length', '단편 소설')}\n"
        f"지정된 합성 문체: {session.get('auto_style', '성애나')}\n"
        f"전체 기획 시놉시스 원문: {session.get('synopsis', '')}\n\n"
        f"수행할 임무: {step_instruction[step]}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o", max_completion_tokens=4000,
        messages=[{"role": "system", "content": AUTO_ENGINE_PROMPT}, {"role": "user", "content": user_content}]
    )
    return response.choices[0].message.content.strip().replace("**", "")

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="일상의 문학", page_icon="✦", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Serif KR', Georgia, serif; }
    .stApp { background-color: #FAF8F5; }
    [data-testid="stSidebar"] { background-color: #F0EDE8 !important; border-right: 1px solid #D4C8BE; }
    .sidebar-title { font-size: 1.1rem; font-weight: 500; color: #2C2C2C; letter-spacing: 0.1em; margin-bottom: 0.2rem; }
    .sidebar-sub { font-size: 0.72rem; color: #9A8880; letter-spacing: 0.2em; margin-bottom: 1.5rem; }
    .session-item { padding: 0.55rem 0.8rem; border-radius: 2px; cursor: pointer; margin-bottom: 0.3rem; border-left: 2px solid transparent; }
    .main-inner { max-width: 760px; margin: 0 auto; padding: 2rem 1rem 4rem 1rem; }
    .session-header { border-bottom: 1px solid #D4C8BE; padding-bottom: 1rem; margin-bottom: 2rem; }
    .session-title-text { font-size: 1.6rem; font-weight: 300; color: #2C2C2C; letter-spacing: 0.12em; margin: 0; }
    .stButton > button { font-family: 'Noto Serif KR', Georgia, serif !important; background-color: #2C2C2C !important; color: #FAF8F5 !important; border: none !important; border-radius: 2px !important; font-size: 0.8rem !important; letter-spacing: 0.2em !important; padding: 0.6rem 1.5rem !important; width: 100%; }
    .stButton > button:hover { background-color: #8B6F5E !important; }
    .stTextArea textarea { font-family: 'Noto Serif KR', Georgia, serif !important; font-size: 0.95rem !important; line-height: 1.85 !important; color: #2C2C2C !important; border: 1px solid #D4C8BE !important; padding: 1rem 1.2rem !important; }
    .story-box { background: #FFFFFF; border: 1px solid #D4C8BE; padding: 2.5rem 2.8rem; line-height: 2.3; font-size: 1.05rem; color: #2C2C2C; white-space: pre-wrap; }
    .auto-box { background-color: #FDFBF7; border: 1px solid #E6DFD5; border-radius: 4px; padding: 1.5rem; margin-bottom: 1.5rem; line-height: 1.9; font-size: 0.98rem; color: #333333; }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────

if "sessions" not in st.session_state:
    _saved, _src = load_data()
    st.session_state.sessions = _saved.get("sessions", [])
    st.session_state.active_idx = _saved.get("active_idx")
    st.session_state.pen_name = _saved.get("pen_name", "")
    st.session_state["_last_saved_at"] = _saved.get("saved_at", "")

if "creating_session" not in st.session_state: st.session_state.creating_session = False

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-title">일상의 문학</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">나의 문학 노트</p>', unsafe_allow_html=True)

    env_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_key:
        api_key_val = st.text_input("API Key", value=st.session_state.get("_api_key_input", ""), type="password", placeholder="sk-...", label_visibility="collapsed")
        if api_key_val != st.session_state.get("_api_key_input", ""):
            st.session_state["_api_key_input"] = api_key_val
            st.rerun()

    if st.button("＋  새 주제 만들기", use_container_width=True, key="new_topic_btn"):
        st.session_state.creating_session = True
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.sessions:
        for idx_real, sess in enumerate(st.session_state.sessions):
            is_active = idx_real == st.session_state.active_idx
            bg = "background:#F5F0EB; border-left:2px solid #8B6F5E;" if is_active else "border-left:2px solid transparent;"
            st.markdown(f'<div class="session-item" style="{bg} padding:0.5rem;"><span style="font-size:0.88rem;">{sess["title"]}</span></div>', unsafe_allow_html=True)
            if st.button("집필실 열기", key=f"open_btn_{sess['id']}", use_container_width=True):
                st.session_state.active_idx = idx_real
                st.session_state.creating_session = False
                st.rerun()

# ─── Main content ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-inner">', unsafe_allow_html=True)

if st.session_state.creating_session:
    st.markdown('<div class="session-header"><p class="session-title-text">새 주제 만들기</p></div>', unsafe_allow_html=True)
    new_title = st.text_input("주제 제목 입력", placeholder="예: 에덴트리...")
    if st.button("✦ 이 주제로 집필실 개방", use_container_width=True):
        title_text = new_title.strip() if new_title.strip() else "제목 없는 주제"
        sess = make_session(title_text)
        st.session_state.sessions.append(sess)
        st.session_state.active_idx = len(st.session_state.sessions) - 1
        st.session_state.creating_session =
