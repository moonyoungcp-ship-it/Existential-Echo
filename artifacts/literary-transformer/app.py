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

AUTO_ENGINE_PROMPT = """당신은 소설 창작의 전 과정을 완벽하게 통제하는 수석 문학 감독이자 거장 소설가입니다.
작가가 제공한 시놉시스와 인물/배경 설정을 바탕으로, 지정된 분량 호흡에 맞추어 [1단계: 인물 구축], [2단계: 세부 배경 묘사], [3단계: 갈등 및 사건 전개], [4단계: 특정 장면 세부 집필] 단계를 정밀하게 수행해야 합니다.

🚨 [장편 이어 쓰기 전용 서사 통제 규칙]
당신은 전체 줄거리를 요약하여 결말을 성급하게 맺어서는 안 됩니다. 
4단계 집필 명령이 발동되면, 소설의 전체 연대기를 압축하지 말고 작가가 명시한 '현재 집필할 특정 장면과 지침'에만 현미경을 대듯 초점을 맞추십시오. 
이전 장면에 축적된 문장들의 분위기와 인물 관계를 정밀하게 이어받아, 서두르지 않는 깊고 느린 호흡의 산문 문단들을 풍부하게 가공하여 완성해라.

절대 주의 사항:
출력하는 결과물 텍스트 전체에 별표 같은 마크다운 강조 기호를 절대로 섞지 마십시오. 오직 정갈한 순수 문장만 출력해야 합니다."""

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
        "auto_steps": {"1": "", "2": "", "3": ""},
        "auto_length": "장편 소설",
        "auto_style": "성애나",
        "current_scene_instruction": "",
        "scenes": []
    }

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def call_auto_engine_api(session: dict, step: str) -> str:
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    
    step_instruction = {
        "1": "1단계 [인물 구축]: 제공된 소설 시놉시스를 바탕으로 주인공 및 주변 인물들의 성격, 심리적 결함, 모순적 내면을 깊이 있게 분석하고 설정안을 도출해라.",
        "2": f"2단계 [세부 배경 묘사]: 앞서 구축된 인물 정보({session['auto_steps'].get('1', '')})를 참조하여, 그들이 호흡할 공간의 대기, 온도, 습도, 사물의 감각적 풍경을 세밀하게 빌드업해라.",
        "3": f"3단계 [갈등 및 사건 전개]: 앞선 인물과 배경 설정 위에서 시놉시스의 사건이 어떻게 고조되는지 구체적인 서사 갈등 축을 설계해라."
    }
    
    user_content = (
        f"소설 기획 제목: {session['title']}\n"
        f"설정된 소설 분량 규격: {session.get('auto_length', '장편 소설')}\n"
        f"지정된 합성 문체: {session.get('auto_style', '성애나')}\n"
        f"전체 기획 시놉시스 원문: {session.get('synopsis', '')}\n\n"
        f"수행할 임무: {step_instruction[step]}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o", max_completion_tokens=4000,
        messages=[{"role": "system", "content": AUTO_ENGINE_PROMPT}, {"role": "user", "content": user_content}]
    )
    return response.choices[0].message.content.strip().replace("**", "")

def call_scene_generation_api(session: dict) -> str:
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    
    past_manuscript = "\n\n".join(f"[{s['scene_title']}]\n{s['scene_content']}" for s in session.get("scenes", []))
    past_context_line = f"\n\n[이전까지 누적 작성된 소설 원고 본문]:\n{past_manuscript}" if past_manuscript else "\n\n[현재 첫 번째 장면을 시작하는 단계입니다.]"

    user_content = (
        f"소설 기획 제목: {session['title']}\n"
        f"지정된 합성 문체: {session.get('auto_style', '성애나')}\n"
        f"전체 기본 시놉시스: {session.get('synopsis', '')}\n"
        f"1단계 인물 설정 환경: {session['auto_steps'].get('1', '')}\n"
        f"2단계 배경 묘사 환경: {session['auto_steps'].get('2', '')}\n"
        f"3단계 갈등 구조 환경: {session['auto_steps'].get('3', '')}"
        f"{past_context_line}\n\n"
        f"🚨 [이번 차례에 집필할 구체적 장면 지침]:\n{session.get('current_scene_instruction', '')}\n\n"
        f"수행할 임무: 위의 구체적 장면 지침에만 현미경을 들이대고, 결말까지 요약하지 말고, 지정된 문체 톤의 길고 세밀한 장편 호흡 본문 문단(최소 4~5문장 이상)을 작성해라."
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
    .story-box { background: #FFFFFF; border: 1px solid #D4C8BE; padding: 2.5rem 2.8rem; line-height: 2.3; font-size: 1.05rem; color: #2C2C2C; white-space: pre-wrap; margin-bottom: 1.5rem; }
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
if "current_editor_buffer" not in st.session_state: st.session_state.current_editor_buffer = ""
if "delete_confirm_idx" not in st.session_state: st.session_state.delete_confirm_idx = None

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

    if st.button("＋  새 장편 소설 기획", use_container_width=True, key="new_topic_btn"):
        st.session_state.creating_session = True
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 사이드바 중첩 제한(Columns nesting) 규격을 피한 정갈한 세로식 목록 기어 ──
    if st.session_state.sessions:
        for idx_real, sess in enumerate(st.session_state.sessions):
            is_active = idx_real == st.session_state.active_idx
            bg = "background:#F5F0EB; border-left:2px solid #8B6F5E;" if is_active else "border-left:2px solid transparent;"
            
            st.markdown(f'<div class="session-item" style="{bg} padding:0.5rem;"><span style="font-size:0.85rem; font-weight:400; color:#2C2C2C;">{sess["title"]}</span></div>', unsafe_allow_html=True)
            
            # 사이드바에서 중첩 컬럼을 쓰지 않고 안전하게 수직 배치
            if st.button("집필실 열기", key=f"open_btn_{sess['id']}", use_container_width=True):
                st.session_state.active_idx = idx_real
                st.session_state.creating_session = False
                st.session_state.current_editor_buffer = ""
                st.session_state.delete_confirm_idx = None
                st.rerun()
                
            if st.session_state.delete_confirm_idx == idx_real:
                st.markdown('<p style="font-size:0.7rem; color:#A24B4B; margin:0.4rem 0 0.2rem 0; text-align:center; font-weight:500;">🚨 정말 이 서재를 영구 삭제할까요?</p>', unsafe_allow_html=True)
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
                if st.button("방 철거하기", key=f"del_req_{sess['id']}", use_container_width=True):
                    st.session_state.delete_confirm_idx = idx_real
                    st.rerun()
                    
            st.markdown("<div style='border-bottom:1px dashed #D4C8BE; margin: 0.6rem 0;'></div>", unsafe_allow_html=True)

# ─── Main content ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-inner">', unsafe_allow_html=True)

if st.session_state.creating_session:
    st.markdown('<div class="session-header"><p class="session-title-text">새 장편 기획서 개설</p></div>', unsafe_allow_html=True)
    new_title = st.text_input("장편 소설 가제 입력", placeholder="예: 에덴트리...")
    if st.button("✦ 이 소설방 가동하기", use_container_width=True):
        title_text = new_title.strip() if new_title.strip() else "제목 없는 장편 소설"
        sess = make_session(title_text)
        st.session_state.sessions.append(sess)
        st.session_state.active_idx = len(st.session_state.sessions) - 1
        st.session_state.creating_session = False
        save_data()
        st.rerun()

elif st.session_state.active_idx is not None and st.session_state.sessions:
    session = st.session_state.sessions[st.session_state.active_idx]
    
    if "synopsis" not in session: session["synopsis"] = ""
    if "auto_steps" not in session: session["auto_steps"] = {"1": "", "2": "", "3": ""}
    if "auto_length" not in session: session["auto_length"] = "장편 소설"
    if "auto_style" not in session: session["auto_style"] = "성애나"
    if "current_scene_instruction" not in session: session["current_scene_instruction"] = ""
    if "scenes" not in session: session["scenes"] = []

    full_compiled_manuscript = "\n\n".join(f"[{s['scene_title']}]\n{s['scene_content']}" for s in session["scenes"])

    st.markdown(f'<div class="session-header"><p class="session-title-text">장편 집필실: {session["title"]}</p></div>', unsafe_allow_html=True)

    col_save1, col_save2 = st.columns(2)
    with col_save1:
        st.text_area("마우스 전체 선택(Ctrl+A) 누적 원고 대피 상자", value=full_compiled_manuscript if full_compiled_manuscript else "아직 축적된 장면 원고가 없습니다.", height=70, label_visibility="collapsed")
    with col_save2:
        st.download_button("내 컴퓨터로 누적 원고 전권 저장 (.txt)", data=full_compiled_manuscript, file_name=f"{session['title']}_통합원고.txt", disabled=len(session["scenes"]) == 0)

    tab_infra, tab_builder, tab_book = st.tabs(["🏗️ 1단계: 소설 기초 뼈대 구축", "🔍 2단계: 현미경식 장면 이어 쓰기", "📚 3단계: 누적 완성 원고 서재"])

    with tab_infra:
        st.markdown('<p class="section-label">전체 기획 대서사 시놉시스</p>', unsafe_allow_html=True)
        synop_input = st.text_area("시놉시스 기술창", value=session["synopsis"], placeholder="여기에 소설의 전체적인 거대 줄거리와 인물 연대기 흐름을 적어주세요.", height=150, key="synop_area", label_visibility="collapsed")
        if synop_input != session["synopsis"]:
            session["synopsis"] = synop_input
            save_data()

        st.markdown("<br>", unsafe_allow_html=True)
        sty_options = ["성애나", "클레어 키건", "김애란"]
        current_sty_prefix = session["auto_style"].split()[0]
        default_sty_idx = sty_options.index(current_sty_prefix) if current_sty_prefix in sty_options else 0
        chosen_sty = st.radio("장편 문체 톤 제어기", sty_options, index=default_sty_idx, horizontal=True)
        if chosen_sty != session["auto_style"]:
            session["auto_style"] = chosen_sty
            save_data()

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">순차적 인프라 빌드업 단추</p>', unsafe_allow_html=True)

        infra_col1, infra_col2, infra_col3 = st.columns(3)
        with infra_col1:
            if st.button("👥 1단계: 전체 인물 구축", use_container_width=True):
                with st.spinner(" 시놉시스 기반 복합 인물 형상화 중..."):
                    session["auto_steps"]["1"] = call_auto_engine_api(session, "1")
                    save_data(); st.rerun()
        with infra_col2:
            if st.button("🏡 2단계: 감각적 배경 설계", use_container_width=True):
                with st.spinner(" 대기 및 사물의 물리적 무대 설계 중..."):
                    session["auto_steps"]["2"] = call_auto_engine_api(session, "2")
                    save_data(); st.rerun()
        with infra_col3:
            if st.button("🎬 3단계: 갈등 구조 조율", use_container_width=True):
                with st.spinner(" 갈등 축 및 서사 타임라인 조율 중..."):
                    session["auto_steps"]["3"] = call_auto_engine_api(session, "3")
                    save_data(); st.rerun()

        if session["auto_steps"].get("1"):
            st.markdown("<br><p class='section-label'>👥 인물 설정안 변경 (수정 가능)</p>", unsafe_allow_html=True)
            edit_s1 = st.text_area("인물 설정 수정", value=session["auto_steps"]["1"], height=150, key="e_s1", label_visibility="collapsed")
            if st.button("✦ 인물 설정 수정본 반영하기", key="b_s1"):
                session["auto_steps"]["1"] = edit_s1
                save_data(); st.success("인물 설정 데이터베이스가 개정되었습니다."); st.rerun()

        if session["auto_steps"].get("2"):
            st.markdown("<br><p class='section-label'>🏡 배경 묘사 풍경 변경 (수정 가능)</p>", unsafe_allow_html=True)
            edit_s2 = st.text_area("배경 설정 수정", value=session["auto_steps"]["2"], height=150, key="e_s2", label_visibility="collapsed")
            if st.button("✦ 배경 설정 수정본 반영하기", key="b_s2"):
                session["auto_steps"]["2"] = edit_s2
                save_data(); st.success("배경 설정 데이터베이스가 개정되었습니다."); st.rerun()

        if session["auto_steps"].get("3"):
            st.markdown("<br><p class='section-label'>🎬 갈등 타임라인 구조 변경 (수정 가능)</p>", unsafe_allow_html=True)
            edit_s3 = st.text_area("갈등 구조 수정", value=session["auto_steps"]["3"], height=150, key="e_s3", label_visibility="collapsed")
            if st.button("✦ 갈등 구조 수정본 반영하기", key="b_s3"):
                session["auto_steps"]["3"] = edit_s3
                save_data(); st.success("갈등 구조 데이터베이스가 개정되었습니다."); st.rerun()

    with tab_builder:
        st.markdown('<p class="section-label">🎬 이번 차례에 집필할 구체적 장면 설정</p>', unsafe_allow_html=True)
        scene_inst = st.text_area("장면 지침창", value=session["current_scene_instruction"], placeholder="예: [장면 1] 재인이 콜센터 삼백이번 칸막이 방 안에서 도입부 풍경만 세밀하게 서술해라.", height=100, key="scene_inst_area", label_visibility="collapsed")
        if scene_inst != session["current_scene_instruction"]:
            session["current_scene_instruction"] = scene_inst
            save_data()

        next_scene_num = len(session["scenes"]) + 1
        scene_title_input = st.text_input("현재 작성 중인 장면의 소제목 명명", value=f"제 {next_scene_num}장. 새로운 벽돌")

        if st.button("✦ 현미경 작동: 지정된 특정 장면만 장편 호흡으로 추출", use_container_width=True):
            if not session["current_scene_instruction"].strip():
                st.warning("이번 차례에 조명할 장면 지침을 먼저 기술해 주세요.")
            else:
                with st.spinner(" 다음 결말을 요약하지 않고, 오직 이 장면에만 현미경을 대고 문장을 제련하는 중..."):
                    generated_scene_block = call_scene_generation_api(session)
                    st.session_state.current_editor_buffer = generated_scene_block
                    st.rerun()

        if st.session_state.current_editor_buffer:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<p class='section-label'>✒️ AI가 벼려낸 장면 초안 (작가 수동 편집창)</p>", unsafe_allow_html=True)
            
            final_edited_buffer = st.text_area("버퍼 에디터", value=st.session_state.current_editor_buffer, height=400, key="buffer_editor_area", label_visibility="collapsed")
            st.session_state.current_editor_buffer = final_edited_buffer

            if st.button("✦ 이 장면의 온도가 마음에 듭니다. 영구 원고고에 이어 붙여 저장하기", use_container_width=True):
                new_scene_payload = {
                    "scene_title": scene_title_input.strip() if scene_title_input.strip() else f"장면 {next_scene_num}",
                    "scene_content": st.session_state.current_editor_buffer,
                    "created_at": now_str()
                }
                session["scenes"].append(new_scene_payload)
                session["current_scene_instruction"] = ""
                st.session_state.current_editor_buffer = ""
                save_data()
                st.success(f"『{new_scene_payload['scene_title']}』 원고가 통합 대서사 전권에 부드럽게 결합되었습니다.")
                st.rerun()

    with tab_book:
        if not session["scenes"]:
            st.markdown('<p style="color:#B0A49C; font-size:0.88rem; margin:3rem 0; text-align:center;">아직 결합된 벽돌 장면이 없습니다.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='section-label'>📚 현재까지 조립 완료된 총 {len(session['scenes'])}개의 대서사 원고</p>", unsafe_allow_html=True)
            for idx_s, sc in enumerate(session["scenes"]):
                st.markdown(f"### {sc['scene_title']}")
                st.markdown(f'<div class="story-box"><div class="story-body">{sc["scene_content"]}</div></div>', unsafe_allow_html=True)
                
                with st.expander(f"🖋️ {sc['scene_title']} 원고 다시 열어 수동 수정하기"):
                    revised_sc_content = st.text_area("과거장면수정", value=sc["scene_content"], height=250, key=f"rev_sc_{idx_s}", label_visibility="collapsed")
                    if st.button("✦ 이 수정한 내용을 책장에 덮어쓰기", key=f"rev_btn_{idx_s}"):
                        session["scenes"][idx_s]["scene_content"] = revised_sc_content
                        save_data(); st.rerun()
            
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            if st.button("🚨 위험: 가장 마지막에 결합한 장면 원고 1개 제거하기 (실행 취소)", use_container_width=True):
                session["scenes"].pop()
                save_data()
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
