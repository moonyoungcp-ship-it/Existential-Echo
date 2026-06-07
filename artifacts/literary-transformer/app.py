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

def available_backups() -> list[tuple[int, str]]:
    result = []
    for n in range(1, MAX_BACKUPS + 1):
        p = _backup_path(n)
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                result.append((n, d.get("saved_at", "날짜 불명")))
            except Exception:
                result.append((n, "날짜 불명"))
    return result

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

출력 텍스트 내부나 앞뒤에 별표(**) 같은 마크다운 표식이나 자질구레한 기호를 절대 사용하지 마세요. 반드시 순수한 문학 텍스트만 출력하세요."""

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

STYLE_MODE_MAP = {
    "묘사의 정석": "[문체 모드: 묘사의 정석]\n형용사를 절대 사용하지 마세요. 감각으로만 장면을 구성합니다. 구체적인 감각 이미지만으로 분위기를 전달하세요.",
    "단단한 단문": "[문체 모드: 단단한 단문]\n접속사를 사용하지 마세요. 문장은 짧고 명료하게 끊어 칩니다. 짧은 단문의 연속으로 리듬을 만드세요.",
    "운율의 호흡": "[문체 모드: 운율의 호흡]\n문장 길이를 의도적으로 조절해 음악성을 부여하세요. 짧은 문장과 긴 문장을 교차하며 호흡을 만듭니다.",
    "낭독의 리듬": "[문체 모드: 낭독의 리듬]\n문장의 길이에 변화를 주어 리듬감을 살리세요. 통사 구조를 반복하여 낭독 시 운율이 느껴지도록 다듬으세요.",
    "빛과 질감": "[문체 모드: 빛과 질감]\n물리적인 빛과 색, 질감의 변화를 세밀하게 포착하세요. 감정을 직접 말하지 않고 물리적 묘사만 활용합니다.",
    "대기의 무게": "[문체 모드: 대기의 무게]\n주변 대기의 습도, 무게감, 온도를 통해 그 공간의 분위기를 압도적으로 표현하세요.",
    "접사 촬영": "[문체 모드: 접사 촬영]\n낡은 흔적이나 미세한 균열 등 시간의 흔적을 극접사로 포착하여 사물 하나에 온전히 집중하세요."
}

STORY_COMPLETE_PROMPT = """당신은 한국 현대 문학을 대표하는 단편 작가입니다.
사용자가 여러 개의 독립적인 문학적 장면들을 보내면, 이것들을 하나의 유기적인 단편소설로 엮어야 합니다.
반드시 문장 앞뒤나 내부에 별표(**) 같은 마크다운 표식을 절대 사용하지 마세요. 순수한 문학 텍스트만 출력하세요."""

AUTO_ENGINE_PROMPT = """당신은 소설 창작의 전 과정을 완벽하게 통제하는 수석 문학 감독이자 거장 소설가입니다.
작가가 제공한 시놉시스를 바탕으로, 지정된 분량 호흡에 맞추어 [1단계: 인물 구축], [2단계: 세부 배경 묘사], [3단계: 갈등 및 사건 전개], [4단계: 최종 문장화 및 합성] 단계를 정밀하게 수행해야 합니다.

소설 분량별 서사 페이스 조절 규칙:
1. 단편 소설 (단일 사건 중심):
   플롯을 방대하게 확장하지 마십시오. 시놉시스의 핵심 사건과 단일한 시선에 집중합니다. 과거 회상은 배제하고 중심 갈등의 뼈대를 밀도 높게 압축하여 빠르게 빌드업합니다.
2. 중편 소설 (입체적 갈등 구조):
   주인공 주변 인물들과의 관계성을 입체적으로 배치합니다. 갈등이 고조되는 정황과 심리 변화를 서두르지 않고 서너 단락에 걸쳐 차분하게 전개합니다.
3. 장편 소설 (대서사 및 정밀 묘사):
   서사를 절대 서둘러 결론짓지 마십시오. 사건이 일어나기 전, 인물이 처한 방의 온도, 가구의 냄새, 피부에 닿는 계절감, 인물의 아주 사소한 손짓과 깊은 전사(過去)까지 극도로 세밀하고 장엄하게 빌드업하십시오.

문체 합성 규칙:
- 클레어 키건 모드: 감상적 개입을 철저히 차단하고, 사물의 물리적 상태와 절제된 행동만으로 침묵의 서사를 암시합니다.
- 김애란 모드: 일상의 비좁은 틈새에서 돋아나는 감각적이고 아릿한 비유, 동시대적 소외감을 예리하게 포착합니다.
- 성애나 모드: 화자 본연의 묵직한 관조적 어조를 유지하며, 세상의 소음과 침묵을 정갈하고 서늘하게 기록합니다.

절대 주의 사항:
출력하는 결과물 텍스트 전체에 별표(**) 같은 마크다운 강조 기호를 절대로 섞지 마십시오. 오직 정갈한 순수 문장만 출력해야 합니다."""

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

def build_system_prompt(mood, season, time_of_day, style_mode=None) -> str:
    hints = []
    if mood and mood in MOOD_MAP: hints.append(MOOD_MAP[mood])
    if season and season in SEASON_MAP: hints.append(SEASON_MAP[season])
    if time_of_day and time_of_day in TIME_MAP: hints.append(TIME_MAP[time_of_day])
    prompt = BASE_SYSTEM_PROMPT
    if hints: prompt += "\n\n[분위기 힌트]\n" + "\n".join(f"- {h}" for h in hints)
    if style_mode and style_mode in STYLE_MODE_MAP: prompt += "\n\n" + STYLE_MODE_MAP[style_mode]
    return prompt

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
    if exclude_last and entries: entries = entries[:-1]
    entries = entries[-MAX_CONTEXT_ENTRIES:]
    msgs = []
    for e in entries:
        msgs.append({"role": "user", "content": e["input"]})
        msgs.append({"role": "assistant", "content": e["output"]})
    return msgs

def call_api(user_text: str, session: dict, is_retry: bool = False, style_override: str | None = "_unset") -> str:
    n = len(user_text.strip())
    effective_style = session.get("style_mode") if style_override == "_unset" else style_override
    system_prompt = build_system_prompt(
        session.get("mood"), session.get("season"), session.get("time_of_day"), effective_style
    )
    length_hint = get_length_hint(n)
    max_tokens = get_length_tokens(n)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(session_chat_history(session, exclude_last=is_retry))
    content = f"[길이 지침: {length_hint}]\n\n{user_text.strip()}"
    if is_retry: content = f"[재시도 지침]\n\n{content}"
    messages.append({"role": "user", "content": content})
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    response = client.chat.completions.create(model="gpt-4o", max_completion_tokens=max_tokens, messages=messages)
    return response.choices[0].message.content.strip().replace("**", "")

def call_auto_engine_api(session: dict, step: str) -> str:
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    
    step_instruction = {
        "1": "1단계 [인물 구축]: 제공된 소설 시놉시스를 바탕으로 주인공 및 주변 인물들의 성격, 심리적 결함, 모순적 내면을 깊이 있게 분석하고 설정안을 도출해라.",
        "2": f"2단계 [세부 배경 묘사]: 앞서 구축된 인물 정보({session['auto_steps']['1']})를 참조하여, 그들이 호흡할 공간의 대기, 온도, 습도, 사물의 감각적 풍경을 세밀하게 빌드업해라.",
        "3": f"3단계 [갈등 및 사건 전개]: 앞선 인물과 배경 설정({session['auto_steps']['2']}) 위에서 시놉시스의 사건이 어떻게 고조되는지 구체적인 서사 타임라인และ 심리적 균열을 설계해라.",
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
        model="gpt-4o",
        max_completion_tokens=4000,
        messages=[
            {"role": "system", "content": AUTO_ENGINE_PROMPT},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content.strip().replace("**", "")

def make_share_card(story_title: str, story_body: str, pen_name: str = "") -> str:
    today = date.today().strftime("%Y.%m.%d")
    border = "─" * 28
    byline = f"  {pen_name}  ·  {today}" if pen_name.strip() else f"  {today}"
    return f"{border}\n  {story_title}\n{border}\n\n{story_body}\n\n{border}\n{byline}\n  일상의 문학\n{border}"

def call_story_complete_api(session: dict) -> str:
    entries = session["entries"]
    scenes = "\n\n---\n\n".join(e["output"] for e in entries)
    hint_parts = [h for h in [session.get("mood"), session.get("season"), session.get("time_of_day")] if h]
    hint_line = ("분위기 힌트: " + " · ".join(hint_parts) + "\n\n") if hint_parts else ""
    user_content = f"주제명: {session['title']}\n\n{hint_line}장면들:\n\n{scenes}"
    client = _get_client()
    if client is None: raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
    response = client.chat.completions.create(
        model="gpt-4o", max_completion_tokens=8192,
        messages=[{"role": "system", "content": STORY_COMPLETE_PROMPT}, {"role": "user", "content": user_content}]
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
    .session-meta-row { font-size: 0.72rem; color: #9A8880; letter-spacing: 0.2em; margin-top: 0.4rem; }
    .entry-block { margin-bottom: 2rem; }
    .entry-output { line-height: 2.1; font-size: 1.02rem; color: #2C2C2C; font-weight: 300; letter-spacing: 0.02em; white-space: pre-wrap; }
    .entry-source { font-size: 0.74rem; color: #B0A49C; letter-spacing: 0.05em; margin-top: 0.5rem; font-style: italic; border-left: 2px solid #E0D8D2; padding-left: 0.7rem; }
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
    st.session_state["_restore_notice"] = ""

if "creating_session" not in st.session_state: st.session_state.creating_session = False
if "last_result" not in st.session_state: st.session_state.last_result = ""
if "_restore_target" not in st.session_state: st.session_state["_restore_target"] = 0

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
        st.session_state.last_result = ""

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
    new_title = st.text_input("주제 제목 입력", placeholder="예: 에덴트리, 출퇴근길의 기억...")
    
    if st.button("✦ 이 주제로 집필실 개방", use_container_width=True):
        title_text = new_title.strip() if new_title.strip() else "제목 없는 주제"
        sess = make_session(title_text)
        st.session_state.sessions.append(sess)
        st.session_state.active_idx = len(st.session_state.sessions) - 1
        st.session_state.creating_session = False
        save_data()
        st.rerun()

elif st.session_state.active_idx is not None and st.session_state.sessions:
    session = st.session_state.sessions[st.session_state.active_idx]
    
    if "synopsis" not in session: session["synopsis"] = ""
    if "auto_steps" not in session: session["auto_steps"] = {"1": "", "2": "", "3": "", "4": ""}
    if "auto_length" not in session: session["auto_length"] = "단편 소설"
    if "auto_style" not in session: session["auto_style"] = "성애나"

    st.markdown(f'<div class="session-header"><p class="session-title-text">{session["title"]}</p></div>', unsafe_allow_html=True)

    col_save1, col_save2 = st.columns(2)
    with col_save1:
        st.text_area("마우스 Ctrl+A 복사용 텍스트 상자", value=session["auto_steps"]["4"] if session["auto_steps"]["4"] else get_full_text(session), height=70, label_visibility="collapsed")
    with col_save2:
        st.download_button("내 컴퓨터로 파일 백업 (.txt)", data=session["auto_steps"]["4"] if session["auto_steps"]["4"] else get_full_text(session), file_name=f"{session['title']}_원고.txt")

    tab_auto, tab_legacy = st.tabs(["✦ 시놉시스 기반 자동화 집필실", "📝 기존 일상 파편 변환기"])

    with tab_auto:
        st.markdown('<p class="section-label">1단계: 기획 시놉시스 기술</p>', unsafe_allow_html=True)
        synop_input = st.text_area("시놉시스 입력창", value=session["synopsis"], placeholder="여기에 소설의 기획의도나 시놉시스, 대략적인 전개 방향을 적어주세요.", height=150, key="synop_area", label_visibility="collapsed")
        if synop_input != session["synopsis"]:
            session["synopsis"] = synop_input
            save_data()

        st.markdown("<br>", unsafe_allow_html=True)
        col_ctrl1, col_ctrl2 = st.columns(2)
        
        # 안전한 매칭을 위한 인덱스 계산부 예외 예방 조치
        len_options = ["단편 소설", "중편 소설", "장편 소설"]
        current_len_prefix = session["auto_length"].split()[0]
        default_len_idx = len_options.index(current_len_prefix) if current_len_prefix in len_options else 0
        
        sty_options = ["성애나", "클레어 키건", "김애란"]
        current_sty_prefix = session["auto_style"].split()[0]
        default_sty_idx = sty_options.index(current_sty_prefix) if current_sty_prefix in sty_options else 0

        with col_ctrl1:
            chosen_len = st.radio("소설 분량 규격 조절 기어", len_options, index=default_len_idx)
            session["auto_length"] = chosen_len
        with col_ctrl2:
            chosen_sty = st.radio("합성 목표 문체 선택", sty_options, index=default_sty_idx)
            session["auto_style"] = chosen_sty

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">단계별 순차적 빌드업 파이프라인</p>', unsafe_allow_html=True)

        col_step1, col_step2, col_step3, col_step4 = st.columns(4)
        
        with col_step1:
            if st.button("1단계: 인물 구축", use_container_width=True):
                with st.spinner(" 인물 분석 중..."):
                    session["auto_steps"]["1"] = call_auto_engine_api(session, "1")
                    save_data()
                    st.rerun()
        with col_step2:
            if st.button("2단계: 배경 묘사", use_container_width=True):
                with st.spinner(" 감각적 무대 설계 중..."):
                    session["auto_steps"]["2"] = call_auto_engine_api(session, "2")
                    save_data()
                    st.rerun()
        with col_step3:
            if st.button("3단계: 갈등 전개", use_container_width=True):
                with st.spinner(" 갈등 타임라인 직조 중..."):
                    session["auto_steps"]["3"] = call_auto_engine_api(session, "3")
                    save_data()
                    st.rerun()
        with col_step4:
            if st.button("4단계: 최종 소설화", use_container_width=True):
                with st.spinner(" 문체 합성 및 원고 집필 중..."):
                    session["auto_steps"]["4"] = call_auto_engine_api(session, "4")
                    save_data()
                    st.rerun()

        if session["auto_steps"]["1"]:
            with st.expander("인물 분석 설정안 확인"): st.markdown(f'<div class="auto-box">{session["auto_steps"]["1"]}</div>', unsafe_allow_html=True)
        if session["auto_steps"]["2"]:
            with st.expander("세부 배경 묘사 풍경 확인"): st.markdown(f'<div class="auto-box">{session["auto_steps"]["2"]}</div>', unsafe_allow_html=True)
        if session["auto_steps"]["3"]:
            with st.expander("사건 전개 및 갈등 축 확인"): st.markdown(f'<div class="auto-box">{session["auto_steps"]["3"]}</div>', unsafe_allow_html=True)
        
        if session["auto_steps"]["4"]:
            st.markdown("<br><p class='section-label'>✦ 최종 완성된 소설 원고 본문</p>", unsafe_allow_html=True)
            st.markdown(f'<div class="story-box"><div class="story-body">{session["auto_steps"]["4"]}</div></div>', unsafe_allow_html=True)

    with tab_legacy:
        if session["entries"]:
            for i, entry in enumerate(session["entries"]):
                st.markdown(f'<div class="entry-block"><div class="entry-output">{entry["output"]}</div><div class="entry-source">원문 파편: {entry["input"]}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        legacy_input = st.text_area("일상의 파편 받아적기", placeholder="오늘 있었던 일을 자유롭게 적어주시면 단편 소설의 파편으로 바꿉니다...", height=100, key="legacy_area", label_visibility="collapsed")
        
        if st.button("✦ 문학적 단락으로 가공하여 추가", use_container_width=True):
            if legacy_input.strip():
                with st.spinner(" 문장 제련 중..."):
                    result = call_api(legacy_input, session, is_retry=False)
                    session["entries"].append({"input": legacy_input.strip(), "output": result, "created_at": now_str()})
                    save_data()
                    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
