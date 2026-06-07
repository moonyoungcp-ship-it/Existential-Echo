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
