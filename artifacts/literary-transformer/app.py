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

AUTO_ENGINE_PROMPT = """당신은 한국 현대 문학의 정수를 담아내는 거장 소설가이자, 창작의 전 과정을 정밀하게 통제하는 수석 문학 감독입니다.

🚨 [최상의 통합 거장 문체 톤 지침]
당신은 한강의 실존적 정적과 깊은 고독, 김애란의 일상적 균열과 아릿한 비유, 편혜영의 내면적 긴장감, 그리고 클레어 키건의 감정적 절제미를 유기적으로 융합한 '최상의 현대 문학 톤'으로만 서술해야 합니다.
- 감정을 슬프다, 외롭다 같은 직접적인 단어로 기술하지 마십시오. 오직 물리적인 빛의 기울기, 사물의 마모된 흔적, 공간의 습도와 공기의 냄새를 통해 내면의 서정을 간접적으로 증명하십시오.
- 서사를 성급하게 요약하여 결말을 맺지 마십시오. 현미경을 대듯 정밀하게 그 장면의 공기를 포착하십시오.
- 문장 내부나 앞뒤에 별표(**) 같은 마크다운 강조 기호를 절대로 섞지 마십시오. 반드시 정갈한 순수 문장만 출력해야 합니다."""

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
        "narrative_pov": "1인칭 화자 ('나')",
        "narrative_tense": "현재제",
        "era_setting": "2000년대 초반",
        "dialect_setting": "사용 안 함 (표준어)",
        "current_scene_instruction": "",
        "scenes": []
    }

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def call_auto_engine_api(session: dict, step: str) -> str:
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    
    step_instruction = {
        "1": f"1단계 [인물 구축]: 제공된 소설 시놉시스를 바탕으로 주인공 및 주변 인물들의 성격, 심리적 결함, 모순적 내면을 깊이 있게 분석하고 설정안을 도출해라. 시대 배경 규격인 [{session.get('era_setting', '2000년대 초반')}]의 사회상과 인물들의 전사가 자연스럽게 녹아들도록 인물을 설계해라.",
        "2": f"2단계 [세부 배경 묘사]: 앞서 구축된 인물 정보({session['auto_steps'].get('1', '')})를 참조하여, 그들이 호흡할 공간의 대기, 온도, 습도, 사물의 감각적 풍경을 세밀하게 빌드업해라. 시대적 배경 고증과 공간의 물리적 질감을 극대화해라.",
        "3": f"3단계 [갈등 및 사건 전개]: 앞선 인물과 배경 설정을 위에서 시놉시스의 사건이 어떻게 균열을 일으키는지 구체적인 서사 갈등 축과 타임라인을 설계해라."
    }
    
    user_content = (
        f"소설 기획 제목: {session['title']}\n"
        f"설정된 소설 시대 배경: {session.get('era_setting', '2000년대 초반')}\n"
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

    # 엄격한 인칭, 시제, 사투리, 시대 제어 지침 바인딩
    string_rules = (
        f"1. 인칭 규격: 반드시 [{session.get('narrative_pov', '1인칭 화자')}] 시점으로 집필하십시오.\n"
        f"2. 시제 규격: 서사의 주된 문장은 반드시 [{session.get('narrative_tense', '현재제')}] 시제를 사용하여 화자의 현재성을 살리십시오.\n"
        f"3. 시대 고증 규격: 이 소설의 시간적 배경은 [{session.get('era_setting', '2000년대 초반')}]입니다. 풍경, 사물, 기술적 수준 등 시대적 현실 고증을 철저히 반영하십시오.\n"
        f"4. 대화체 사투리 규격: 인물들이 나누는 대화나 독백의 구어체에는 반드시 [{session.get('dialect_setting', '사용 안 함')}]의 억양과 고유 방언을 사실적으로 반영하되, 지문은 정갈한 문학 톤을 유지하십시오."
    )

    user_content = (
        f"소설 기획 제목: {session['title']}\n"
        f"전체 기본 시놉시스: {session.get('synopsis', '')}\n"
        f"1단계 인물 설정 환경: {session['auto_steps'].get('1', '')}\n"
        f"2단계 배경 묘사 환경: {session['auto_steps'].get('2', '')}\n"
        f"3단계 갈등 구조 환경: {session['auto_steps'].get('3', '')}"
        f"{past_context_line}\n\n"
        f"🚨 [엄격 준수해야 할 소설 집필 규격]:\n{string_rules}\n\n"
        f"🚨 [이번 차례에 집필할 구체적 장면 지침]:\n{session.get('current_scene_instruction', '')}\n\n"
        f"수행할 임무: 위의 모든 문학적 규격과 장면 지침에만 현미경을 들이대고, 결말까지 요약하지 말고, 서두르지 않는 거장의 깊은 호흡으로 생생한 장편 본문 문단을 작성해라."
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
                st.markdown('<p style="font-size:0.7rem; color:#A24B4B; margin:0.4rem 0 0.2rem 0; text-align:center; font-weight:500;">🚨 진짜 이 서재를 영구 삭제할까요?</p>', unsafe_allow_html=True)
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
    
    # 세부 옵션 키 마이그레이션 안전 점검
    if "synopsis" not in session: session["synopsis"] = ""
    if "auto_steps" not in session: session["auto_steps"] = {"1": "", "2": "", "3": ""}
    if "narrative_pov" not in session: session["narrative_pov"] = "1인칭 화자 ('나')"
    if "narrative_tense" not in session: session["narrative_tense"] = "현재형"
    if "era_setting" not in session: session["era_setting"] = "2000년대 초반"
    if "dialect_setting" not in session: session["dialect_setting"] = "사용 안 함 (표준어)"
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

    # ==================== 탭 1: 소설 기초 뼈대 구축 ====================
    with tab_infra:
        st.markdown('<p class="section-label">전체 기획 대서사 시놉시스</p>', unsafe_allow_html=True)
        synop_input = st.text_area("시놉시스 기술창", value=session["synopsis"], placeholder="여기에 소설의 전체적인 거대 줄거리와 인물 연대기 흐름을 적어주세요.", height=150, key="synop_area", label_visibility="collapsed")
        if synop_input != session["synopsis"]:
            session["synopsis"] = synop_input
            save_data()

        st.markdown("<br><p class='section-label'>🚨 문학 규격 설정 기어 (장편 전용)</p>", unsafe_allow_html=True)
        
        # 작가 선택 버튼을 완벽히 도려내고 인칭, 시제, 사투리, 시대 제어판으로 리모델링
        infra_ctrl1, infra_ctrl2 = st.columns(2)
        with infra_ctrl1:
            pov_options = ["1인칭 화자 ('나')", "3인칭 전지적 시점", "3인칭 제한적 관찰자 시점"]
            default_pov_idx = pov_options.index(session["narrative_pov"]) if session["narrative_pov"] in pov_options else 0
            chosen_pov = st.radio("소설 서사 인칭 선택", pov_options, index=default_pov_idx)
            session["narrative_pov"] = chosen_pov
            
            tense_options = ["현재형 (생동감과 깊은 실존적 몰입)", "과거형 (전통적 산문의 안정된 서사 호흡)"]
            default_tense_idx = 0 if "현재" in session["narrative_tense"] else 1
            chosen_tense = st.radio("서사 주안 시제 설정", tense_options, index=default_tense_idx)
            session["narrative_tense"] = chosen_tense
            
        with infra_ctrl2:
            era_input = st.text_input("시대적 배경 역사 고증 설정 (텍스트 입력)", value=session["era_setting"], placeholder="예: 1990년대 중반 IMF 직전, 2000년대 초반 콜센터...")
            session["era_setting"] = era_input
            
            dialect_options = ["사용 안 함 (표준어 중심)", "제주 방언 (제주 사투리)", "경상 방언", "전라 방언", "충청 방언"]
            default_dia_idx = dialect_options.index(session["dialect_setting"]) if session["dialect_setting"] in dialect_options else 0
            chosen_dia = st.radio("대화체 사투리 고증 기어", dialect_options, index=default_dia_idx)
            session["dialect_setting"] = chosen_dia
            
        save_data()

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">순차적 인프라 빌드업 단추</p>', unsafe_allow_html=True)

        infra_col1, infra_col2, infra_col3 = st.columns(3)
        with infra_col1:
            if st.button("👥 1단계: 전체 인물 구축", use_container_width=True):
                with st.spinner(" 시놉시스 및 시대 고증 기반 복합 인물 형상화 중..."):
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

    # ==================== 탭 2: 현미경식 장면 이어 쓰기 ====================
    with tab_builder:
        st.markdown('<p class="section-label">🎬 이번 차례에 집필할 구체적 장면 설정</p>', unsafe_allow_html=True)
        scene_inst = st.text_area("장면 지침창", value=session["current_scene_instruction"], placeholder="예: [장면 1] 재인이 서늘한 콜센터 삼백이번 칸막이 방 안에서 도입부 풍경만 세밀하게 서술해라.", height=100, key="scene_inst_area", label_visibility="collapsed")
        if scene_inst != session["current_scene_instruction"]:
            session["current_scene_instruction"] = scene_inst
            save_data()

        next_scene_num = len(session["scenes"]) + 1
        scene_title_input = st.text_input("현재 작성 중인 장면의 소제목 명명", value=f"제 {next_scene_num}장. 새로운 벽돌")

        if st.button("✦ 현미경 작동: 지정된 특정 장면만 장편 호흡으로 추출", use_container_width=True):
            if not session["current_scene_instruction"].strip():
                st.warning("이번 차례에 조명할 장면 지침을 먼저 기술해 주세요.")
            else:
                with st.spinner(" 설정된 문학 기어와 고증을 동기화하여 장면을 집필하는 중..."):
                    generated_scene_block = call_scene_generation_api(session)
                    st.session_state.current_editor_buffer = generated_scene_block
                    st.rerun()

        if st.session_state.current_editor_buffer:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<p class='section-label'>✒️ 통합 거장 톤으로 제련된 장면 초안 (수동 편집 및 다듬기)</p>", unsafe_allow_html=True)
            
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
                st.success(f"『{new_scene_payload['scene_title']}』 원고가 장편 통합 서사에 완벽하게 결합되었습니다.")
                st.rerun()

    # ==================== 탭 3: 누적 완성 원고 서재 ====================
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
