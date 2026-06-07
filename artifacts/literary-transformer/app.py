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

# ─── Prompts ──────────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """당신은 한국 현대 문학의 정수를 담아내는 거장 소설가이자, 연작소설의 유기적 흐름을 통제하는 문학 감독입니다.

🚨 [최상의 거장 문체 톤 및 연작 제련 지침]
당신은 한강의 실존적 정적, 김애란의 일상적 균열과 아릿한 비유, 클레어 키건의 절제미를 유기적으로 융합한 최상의 현대 문학 문체로 산문을 제련해야 합니다.
- 독립된 단편 소설 규격(원고지 70~100장 호흡)의 리듬을 지키십시오. 전체 이야기를 요약하여 결말을 성급하게 맺지 마십시오.
- 감정을 직접적인 단어로 설명하지 말고, 사물의 마모된 흔적, 물리적인 빛의 기울기, 공간의 온도와 공기의 습도를 통해 인물의 내면을 관조적으로 보여주십시오.
- 대화나 독백의 구어체에는 설정된 사투리의 고유 억양을 사실적으로 녹여내되, 지문은 서늘하고 정갈한 문학 톤을 엄격하게 유지하십시오.
- 문장 내부에 별표(**) 같은 마크다운 표식을 절대로 사용하지 마십시오. 오직 정갈한 순수 문장만 출력해야 합니다."""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_session(title: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "entries": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "synopsis": "", # 연작소설 전체 세계관 기획의도
        "narrative_pov": "1인칭 화자 ('나')",
        "narrative_tense": "현재형",
        "era_setting": "2000년대 초반",
        "dialect_setting": "사용 안 함 (표준어 중심)",
        "current_scene_instruction": "",
        "scenes": [] # 완성하여 책장에 꽂은 독립 단편 소설들의 목록
    }

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def call_scene_generation_api(session: dict) -> str:
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    
    # 이미 책장에 꽂힌 이전 단편들의 맥락 상속
    past_manuscript = "\n\n".join(f"[{s['scene_title']}]\n{s['scene_content']}" for s in session.get("scenes", []))
    past_context_line = f"\n\n[이미 연작으로 완성되어 책장에 꽂힌 앞선 단편 소설들]:\n{past_manuscript}" if past_manuscript else "\n\n[현재 연작소설의 첫 번째 단편을 시작하는 단계입니다.]"

    string_rules = (
        f"1. 인칭 규격: 반드시 [{session.get('narrative_pov', '1인칭 화자')}] 시점으로 집필하십시오.\n"
        f"2. 시제 규격: 서사의 문장은 반드시 [{session.get('narrative_tense', '현재형')}] 시제를 유지하여 현장감을 극대화하십시오.\n"
        f"3. 시대 고증 규격: 이 단편의 구체적 시간 배경은 [{session.get('era_setting', '2000년대 초반')}]입니다. 역사적 현실 고증을 철저히 반영하십시오.\n"
        f"4. 대화체 사투리 규격: 인물들의 대사에는 반드시 [{session.get('dialect_setting', '사용 안 함')}]의 방언을 사실적으로 반영하십시오."
    )

    user_content = (
        f"연작소설 전체 대세계관 기획의도: {session.get('synopsis', '')}\n"
        f"{past_context_line}\n\n"
        f"🚨 [엄격 준수해야 할 소설 규격 지침]:\n{string_rules}\n\n"
        f"🚨 [현재 집필할 단편의 구체적 장면 및 핵심 서사 지침]:\n{session.get('current_scene_instruction', '')}\n\n"
        f"수행할 임무: 앞서 완성된 단편들의 문조와 흐름을 정밀하게 상속받아, 전체 이야기를 압축 요약하지 말고 오직 이번 단편의 지침에만 현미경을 대고 단편소설 고유의 밀도 높은 호흡(4~5문장 이상의 문단들)으로 본문을 집필해라."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o", max_completion_tokens=4000,
        messages=[{"role": "system", "content": BASE_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    )
    return response.choices[0].message.content.strip().replace("**", "")

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="일상의 문학", page_icon="✦", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght=300;400;500;600&display=swap');
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
    .story-box { background: #FFFFFF; border: 1px solid #D4C8BE; padding: 2.5rem 2.8rem; line-height: 2.3; font-size: 1.05rem; color: #2C2C2C; white-space: pre-wrap; margin-bottom: 1.5rem; }
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
if "current_editor_buffer" not in st.session_state: st.session_state.current_editor_buffer = ""
if "delete_confirm_idx" not in st.session_state: st.session_state.delete_confirm_idx = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-title">일상의 문학</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">연작소설 대서제</p>', unsafe_allow_html=True)

    env_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_key:
        api_key_val = st.text_input("API Key", value=st.session_state.get("_api_key_input", ""), type="password", placeholder="sk-...", label_visibility="collapsed")
        if api_key_val != st.session_state.get("_api_key_input", ""):
            st.session_state["_api_key_input"] = api_key_val
            st.rerun()

    if st.button("＋  새 연작 기획서 개설", use_container_width=True, key="new_topic_btn"):
        st.session_state.creating_session = True
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.sessions:
        for idx_real, sess in enumerate(st.session_state.sessions):
            is_active = idx_real == st.session_state.active_idx
            bg = "background:#F5F0EB; border-left:2px solid #8B6F5E;" if is_active else "border-left:2px solid transparent;"
            
            session_time = sess.get("created_at", "날짜 불명")
            
            st.markdown(
                f'<div class="session-item" style="{bg} padding:0.5rem 0.8rem;">'
                f'<div style="font-size:0.85rem; font-weight:400; color:#2C2C2C; margin-bottom:0.15rem;">{sess["title"]}</div>'
                f'<div style="font-size:0.68rem; color:#9A8880; letter-spacing:0.02em;">📅 {session_time}</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            if st.button("집필실 열기", key=f"open_btn_{sess['id']}", use_container_width=True):
                st.session_state.active_idx = idx_real
                st.session_state.creating_session = False
                st.session_state.current_editor_buffer = ""
                st.session_state.delete_confirm_idx = None
                st.rerun()
                
            if st.session_state.delete_confirm_idx == idx_real:
                st.markdown('<p style="font-size:0.7rem; color:#A24B4B; margin:0.4rem 0 0.2rem 0; text-align:center; font-weight:500;">🚨 진짜 이 연작을 영구 삭제할까요?</p>', unsafe_allow_html=True)
                if st.button("🗑️ 네, 흔적 없이 삭제합니다", key=f"del_yes_{sess['id']}", use_container_width=True):
                    st.session_state.sessions.pop(idx_real)
                    if st.session_state.active_idx == idx_real:
                        st.session_state.active_idx = 0 if st.session_state.sessions else None
                    elif st.session_state.active_idx is not None and st.session_state.active_idx > idx_real:
                        st.session_state.active_idx -= 1
                    st.session_state.delete_confirm_idx = None
                    save_data()
                    st.rerun()
                if st.button("❌ 아니오, 유지합니다", key=f"del_no_{sess['id']}", use_container_width=True):
                    st.session_state.delete_confirm_idx = None
                    st.rerun()
            else:
                if st.button("연작 철거하기", key=f"del_req_{sess['id']}", use_container_width=True):
                    st.session_state.delete_confirm_idx = idx_real
                    st.rerun()
                    
            st.markdown("<div style='border-bottom:1px dashed #D4C8BE; margin: 0.6rem 0;'></div>", unsafe_allow_html=True)

# ─── Main content ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-inner">', unsafe_allow_html=True)

if st.session_state.creating_session:
    st.markdown('<div class="session-header"><p class="session-title-text">새 연작 기획서 개설</p></div>', unsafe_allow_html=True)
    new_title = st.text_input("연작소설 전체 이름 (가제)", placeholder="예: 에덴트리...")
    if st.button("✦ 이 연작 작업실 가동하기", use_container_width=True):
        title_text = new_title.strip() if new_title.strip() else "제목 없는 연작 대서사"
        sess = make_session(title_text)
        st.session_state.sessions.append(sess)
        st.session_state.active_idx = len(st.session_state.sessions) - 1
        st.session_state.creating_session = False
        save_data()
        st.rerun()

elif st.session_state.active_idx is not None and st.session_state.sessions:
    session = st.session_state.sessions[st.session_state.active_idx]
    
    if "synopsis" not in session: session["synopsis"] = ""
    if "narrative_pov" not in session: session["narrative_pov"] = "1인칭 화자 ('나')"
    if "narrative_tense" not in session: session["narrative_tense"] = "현재형"
    if "era_setting" not in session: session["era_setting"] = "2000년대 초반"
    if "dialect_setting" not in session: session["dialect_setting"] = "사용 안 함 (표준어 중심)"
    if "current_scene_instruction" not in session: session["current_scene_instruction"] = ""
    if "scenes" not in session: session["scenes"] = []

    full_compiled_manuscript = "\n\n\n".join(f"【 {s['scene_title']} 】\n\
