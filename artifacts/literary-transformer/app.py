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

LONG_NARRATIVE_PROMPT = """당신은 한국 현대 문학의 정수를 담아내는 거장 소설가이자, 원고의 서사적 호흡을 가장 정밀하고 깊게 통제하는 수석 문학 감독입니다.

🚨 [최상의 거장 문체 톤 및 장편 분량 확장 지침]
당신은 한강의 실존적 정적과 고독, 김애란의 일상적 균열과 아릿한 비유, 클레어 키건의 감정적 절제미를 유기적으로 융합한 최상의 현대 문학 톤으로만 서술해야 합니다.

1. 분량과 호흡의 극대화 규격: 
   - 서사를 절대로 요약하거나 상황을 성급하게 결론짓지 마십시오. 아주 미세한 순간(예: 의자에서 일어나는 동작, 종이를 뒤적이는 손길)에 현미경을 들이대고 그 과정 전체를 정밀하게 복원하여 출력 분량을 최대한 길고 풍성하게 확보하십시오.
   - 하나의 문단은 최소 5~7문장 이상의 긴 산문 호흡으로 구성되어야 하며, 전체 본문은 서두르지 않는 거장의 호흡으로 밀도 높게 채워져야 합니다.

2. 감정의 간접 증명: 
   - 슬프다, 외롭다, 답답하다 같은 직접적인 감정 단어는 단 한 번도 사용하지 마십시오. 오직 물리적인 빛의 기울기, 사물의 마모된 흔적, 공간의 습도와 공기의 냄새, 인물의 미세한 근육 움직임을 통해서만 내면의 서정을 간접적으로 증명하십시오.

3. 엄격한 금지 사항:
   - 문장 내부나 앞뒤에 별표(**) 같은 마크다운 강조 기호를 절대로 섞지 마십시오. 복사 붙여넣기에 방해가 됩니다. 반드시 정갈한 순수 문장만 출력해야 합니다."""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_session(title: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "entries": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "synopsis": "", 
        "narrative_pov": "1인칭 화자 ('나')",
        "narrative_tense": "현재형",
        "era_setting": "2000년대 초반",
        "dialect_setting": "사용 안 함 (표준어 중심)",
        "current_scene_instruction": "",
        "scenes": [] 
    }

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def call_scene_generation_api(session: dict) -> str:
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    
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
        f"수행할 임무: 지정된 장면 지침의 시간이 아주 느리게 흐르도록 현미경 배율을 최대화해라. 줄거리를 요약하여 미래의 사건으로 도망치지 말고, 오직 이 시공간에 머물며 사물과 대기의 질감을 낱낱이 파헤쳐 최소 원고지 15~20장 분량 이상에 달하는 길고 촘촘한 호흡의 문학적 본문을 반환해라."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o", 
        max_completion_tokens=4000, # 최대 출력 토큰을 4000으로 끝까지 개방
        temperature=0.75,          # 서사의 다양성과 긴 묘사를 위해 온도 최적화
        messages=[{"role": "system", "content": LONG_NARRATIVE_PROMPT}, {"role": "user", "content": user_content}]
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

    full_compiled_manuscript = "\n\n\n".join(f"【 {s['scene_title']} 】\n\n{s['scene_content']}" for s in session["scenes"])

    st.markdown(f'<div class="session-header"><p class="session-title-text">연작 집필실: 『{session["title"]}』</p></div>', unsafe_allow_html=True)

    col_save1, col_save2 = st.columns(2)
    with col_save1:
        st.text_area("마우스 전체 선택(Ctrl+A) 연작 전권 대피 상자", value=full_compiled_manuscript if full_compiled_manuscript else "아직 축적된 단편 소설 원고가 없습니다.", height=70, label_visibility="collapsed")
    with col_save2:
        st.download_button("내 컴퓨터로 연작 통합 마스터본 저장 (.txt)", data=full_compiled_manuscript, file_name=f"{session['title']}_연작통합본.txt", disabled=len(session["scenes"]) == 0)

    tab_infra, tab_builder, tab_book = st.tabs(["🏗️ 1단계: 거대 대세계관 설정", "🔍 2단계: 개별 단편 소설 세부 제련", "📚 3단계: 연작 단편집 책장"])

    # ==================== 탭 1: 거대 대세계관 설정 ====================
    with tab_infra:
        st.markdown('<p class="section-label">🌌 연작소설 전체를 관통하는 거대 세계관 및 기획 의도</p>', unsafe_allow_html=True)
        synop_input = st.text_area("세계관 기술창", value=session["synopsis"], placeholder="기계의 장황한 호흡을 이끌어내기 위해, 여기 대세계관 설정 상자는 완전히 비워두거나 단순 가제만 남겨두시는 것을 강력하게 권장합니다.", height=150, key="synop_area", label_visibility="collapsed")
        if synop_input != session["synopsis"]:
            session["synopsis"] = synop_input
            save_data()

        st.markdown("<br><p class='section-label'>🚨 연작 통합 문학 규격 기어</p>", unsafe_allow_html=True)
        
        infra_ctrl1, infra_ctrl2 = st.columns(2)
        with infra_ctrl1:
            pov_options = ["1인칭 화자 ('나')", "3인칭 전지적 시점", "3인칭 제한적 관찰자 시점"]
            default_pov_idx = pov_options.index(session["narrative_pov"]) if session["narrative_pov"] in pov_options else 0
            chosen_pov = st.radio("연작 화자 인칭 선택", pov_options, index=default_pov_idx)
            session["narrative_pov"] = chosen_pov
            
            tense_options = ["현재형 (실존적 몰입과 현재성 극대화)", "과거형 (전통적 단편 산문의 안정된 리듬)"]
            default_tense_idx = 0 if "현재" in session["narrative_tense"] else 1
            chosen_tense = st.radio("연작 시제 호흡 설정", tense_options, index=default_tense_idx)
            session["narrative_tense"] = chosen_tense
            
        with infra_ctrl2:
            era_input = st.text_input("단편별 구체적 시대 배경 고증 역사 설정", value=session["era_setting"], placeholder="예: 2000년대 초반 무더운 7월...")
            session["era_setting"] = era_input
            
            dialect_options = ["사용 안 함 (표준어 중심)", "제주 방언 (제주 사투리)", "경상 방언", "전라 방언", "충청 방언"]
            default_dia_idx = dialect_options.index(session["dialect_setting"]) if session["dialect_setting"] in dialect_options else 0
            chosen_dia = st.radio("연작 대화체 사투리 제어 기어", dialect_options, index=default_dia_idx)
            session["dialect_setting"] = chosen_dia
            
        save_data()

    # ==================== 탭 2: 개별 단편 소설 세부 제련 ====================
    with tab_builder:
        st.markdown('<p class="section-label">🎬 이번 차례에 제련할 단편 소설의 구체적 장면 및 현미경 지침 기술</p>', unsafe_allow_html=True)
        scene_inst = st.text_area("장면 지침창", value=session["current_scene_instruction"], placeholder="여기에 '미래의 이야기(콜센터, 결혼 등)'를 다 빼고, 오직 지금 이 순간의 정물 스케치만 정밀하게 지시해 주셔야 문장 분량이 대폭 확장됩니다.\n\n예: 재인이 신용회복위원회 대기실 의자에 앉아 번호표를 쥐고 있다. 옆자리 남자의 셔츠 깃 모양, 유리창 너머 상담원들의 무미건조한 입 모양, 가방 속에 든 채무 내역서 서류의 차가운 촉감만을 서두르지 말고 아주 장황하고 정밀하게 서술해라. 요약 금지.", height=150, key="scene_inst_area", label_visibility="collapsed")
        if scene_inst != session["current_scene_instruction"]:
            session["current_scene_instruction"] = scene_inst
            save_data()

        next_scene_num = len(session["scenes"]) + 1
        scene_title_input = st.text_input("현재 집필 중인 단편 소설의 독립 가제 명명", value=f"제 {next_scene_num}부. 새로운 파편")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✦ 최상위 거장 톤 가동: 독립 단편 원고 제련", use_container_width=True):
            if not session["current_scene_instruction"].strip():
                st.warning("이번 차례에 온전히 몰입할 단편 소설 지침을 먼저 명시해 주세요.")
            else:
                with st.spinner(" 설정된 문학 기어의 출력을 최대화하여 요약 없는 장산문 원고를 직조하는 중..."):
                    generated_scene_block = call_scene_generation_api(session)
                    st.session_state.current_editor_buffer = generated_scene_block
                    st.rerun()

        if st.session_state.current_editor_buffer:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<p class='section-label'>✒️ 거장 톤으로 제련된 단편 초안 (작가 전면 다듬기 및 가공 편집창)</p>", unsafe_allow_html=True)
            
            final_edited_buffer = st.text_area("버퍼 에디터", value=st.session_state.current_editor_buffer, height=500, key="buffer_editor_area", label_visibility="collapsed")
            st.session_state.current_editor_buffer = final_edited_buffer

            if st.button("✦ 이 단편의 온도가 완벽합니다. 연작 대서사 책장에 결합 저장하기", use_container_width=True):
                new_scene_payload = {
                    "scene_title": scene_title_input.strip() if scene_title_input.strip() else f"제 {next_scene_num}부. 단편",
                    "scene_content": st.session_state.current_editor_buffer,
                    "created_at": now_str()
                }
                session["scenes"].append(new_scene_payload)
                session["current_scene_instruction"] = ""
                st.session_state.current_editor_buffer = ""
                save_data()
                st.success(f"『{new_scene_payload['scene_title']}』 원고가 연작소설 대서사 단편집 책장에 정갈하게 결합되었습니다.")
                st.rerun()

    # ==================== 탭 3: 연작 단편집 책장 ====================
    with tab_book:
        if not session["scenes"]:
            st.markdown('<p style="color:#B0A49C; font-size:0.88rem; margin:3rem 0; text-align:center;">아직 결합되어 꽂힌 단편 소설이 없습니다.<br>2단계 탭에서 연작의 첫 번째 파편을 빌드업하여 서재 책장을 채워나가세요.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='section-label'>📚 현재까지 조립 완료된 총 {len(session['scenes'])}편의 연작 단편 단편집</p>", unsafe_allow_html=True)
            for idx_s, sc in enumerate(session["scenes"]):
                st.markdown(f"### {sc['scene_title']}")
                st.markdown(f'<div class="story-box"><div class="story-body">{sc["scene_content"]}</div></div>', unsafe_allow_html=True)
                
                with st.expander(f"I_I 🖋️ {sc['scene_title']} 단편 다시 열어 전면 수동 수정하기"):
                    revised_sc_content = st.text_area("과거단편수정", value=sc["scene_content"], height=300, key=f"rev_sc_{idx_s}", label_visibility="collapsed")
                    if st.button("✦ 이 고친 단편 내용을 책장에 새롭게 덮어쓰기", key=f"rev_btn_{idx_s}"):
                        session["scenes"][idx_s]["scene_content"] = revised_sc_content
                        save_data(); st.rerun()
            
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            if st.button("🚨 위험: 가장 마지막에 책장에 꽂은 단편 원고 1편 철거하기 (실행 취소)", use_container_width=True):
                session["scenes"].pop()
                save_data()
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
