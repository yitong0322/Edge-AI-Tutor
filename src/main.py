#!/usr/bin/env python3
"""
AI Tutor — 完整版
单击按钮切换 Chat / RAG 模式
长按录音 → ASR → (路由+检索) → LLM → TTS
双击清空历史
"""

import os, time, subprocess, threading, json
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import ollama, textwrap
from PIL import Image, ImageDraw, ImageFont
from WhisPlay import WhisPlayBoard
import RPi.GPIO as GPIO
import traceback
from typing import Any, List
from pydantic import ConfigDict

os.environ['ORT_LOGGING_LEVEL'] = '3'

# ============================================================
# 配置区域
# ============================================================
MODEL_NAME     = "llama3.2-3b-q6:latest"
ASR_URL        = "http://localhost:8000/transcribe"
TTS_SPEAK_URL  = "http://localhost:8001/speak"
TTS_STOP_URL   = "http://localhost:8001/stop"
TTS_HEALTH_URL = "http://localhost:8001/health"
PROJECT_DIR    = "/home/user/AITutor"
WAV_PATH       = os.path.join(PROJECT_DIR, "req.wav")

# RAG 配置（路径与 chatbot_3b_RAG.py / ingest_pdf.py 完全一致）
ROUTES_FILE     = os.path.join(PROJECT_DIR, "routes.json")
VECTOR_DB_PATH  = os.path.join(PROJECT_DIR, "chroma_db")
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"  # fastembed 本地 ONNX 模型，不走 Ollama
COLLECTION_NAME = "electronics_knowledge"
RAG_K           = 1        # 每次检索返回的文档片段数（减少 prefill tokens）
ROUTE_THRESHOLD = 0.35     # 路由相似度门槛

# 录音参数
SAMPLE_RATE    = 48000
MIN_RECORD_SEC = 0.5
LONG_PRESS_SEC = 0.4

# 对话参数
MAX_HISTORY = 6

# LLM 参数
LLM_OPTIONS = {
    'num_thread':     4,
    'num_ctx':        2048,
    'temperature':    0.3,
    'top_p':          0.9,
    'top_k':          40,
    'repeat_penalty': 1.1,
}

RAG_LLM_OPTIONS = LLM_OPTIONS  # RAG 模式复用同一套参数

# TTS 切句参数
STRONG_PUNCT    = set(['.', '!', '?', '。', '！', '？', '\n'])
WEAK_PUNCT      = set([',', ';', ':', '，', '；', '：'])
FORCE_SPLIT_LEN = 60
MIN_SENT_LEN    = 8
FIRST_SENT_MIN  = 20

# 屏幕刷新
SCREEN_INTERVAL = 0.3

# 超时
ASR_TIMEOUT = 20
TTS_TIMEOUT = 2
SVC_TIMEOUT = 2

# ============================================================
# 全局状态
# ============================================================
chat_history = []
stop_event   = threading.Event()
current_mode = 'chat'   # 'chat' 或 'rag'
rag_engine   = None     # RAGEngine 实例（启动时初始化）

# ============================================================
# 声卡检测
# ============================================================
def get_wm8960_index() -> int:
    try:
        for i, dev in enumerate(sd.query_devices()):
            if 'wm8960' in dev['name'].lower() and dev['max_input_channels'] > 0:
                print(f"[Audio] WM8960 found at index {i}: {dev['name']}")
                return i
    except Exception as e:
        print(f"[Audio] Device search error: {e}")
    print("[Audio] WM8960 not found, using default (index 0)")
    return 0

INPUT_DEVICE_ID = get_wm8960_index()

# ============================================================
# FastEmbed 适配层（不走 Ollama，不占 LLM 内存）
# ============================================================
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

class FastEmbedEmbeddings(Embeddings):
    """fastembed → LangChain Embeddings 接口适配器"""
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0].tolist()


# ============================================================
# RAG 引擎
# ============================================================
def _import_rag_libs():
    from semantic_router import Route
    from semantic_router.routers import SemanticRouter
    from semantic_router.encoders import DenseEncoder
    from semantic_router.index.local import LocalIndex
    return Route, SemanticRouter, DenseEncoder, LocalIndex


class RAGEngine:
    """
    Semantic Router + ChromaDB 检索。
    embedding 全部由 fastembed 完成，完全不经过 Ollama。
    """

    def __init__(self):
        print("[RAG] Initializing RAG Engine (fastembed)...")
        self.retriever = None
        self.router    = None

        # 共用一个 fastembed 实例（避免重复加载模型）
        self._embeddings = FastEmbedEmbeddings()

        Route, SemanticRouter, DenseEncoder, LocalIndex = _import_rag_libs()

        # ── 1. 连接向量数据库 ─────────────────────────────
        from langchain_chroma import Chroma
        if os.path.exists(VECTOR_DB_PATH):
            try:
                vectorstore = Chroma(
                    collection_name   = COLLECTION_NAME,
                    embedding_function= self._embeddings,
                    persist_directory = VECTOR_DB_PATH,
                )
                self.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": RAG_K}
                )
                print("[RAG] ChromaDB loaded.")
            except Exception as e:
                print(f"[RAG] ChromaDB Error: {e}")
        else:
            print(f"[RAG] Warning: chroma_db not found at {VECTOR_DB_PATH}")

        # ── 2. 初始化路由 ─────────────────────────────────
        self._load_router(Route, SemanticRouter, DenseEncoder, LocalIndex)

    def _make_encoder(self, DenseEncoder):
        """用 fastembed 实现 semantic-router 的 DenseEncoder 接口"""
        embed_fn = self._embeddings   # 复用已加载的模型

        class FastEncoder(DenseEncoder):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            _embed_fn: Any = None

            def __init__(self_):
                super().__init__(name=EMBED_MODEL)
                self_._embed_fn = embed_fn

            def __call__(self_, docs: List[str]) -> List[List[float]]:
                print(f"[Encoder] Embedding {len(docs)} docs (fastembed)...")
                return self_._embed_fn.embed_documents(docs)

        return FastEncoder()

    def _load_router(self, Route, SemanticRouter, DenseEncoder, LocalIndex):
        if not os.path.exists(ROUTES_FILE):
            print(f"[RAG] Warning: routes.json not found at {ROUTES_FILE}")
            return

        print(f"[RAG] Loading routes from {ROUTES_FILE}...")
        with open(ROUTES_FILE, 'r') as f:
            data = json.load(f)

        routes = [
            Route(
                name           = item['name'],
                utterances     = item['utterances'],
                score_threshold= ROUTE_THRESHOLD,
            )
            for item in data
        ]

        print(f"[RAG] Built {len(routes)} routes. Encoding utterances...")
        try:
            encoder = self._make_encoder(DenseEncoder)

            all_utterances  = []
            all_route_names = []
            for route in routes:
                for u in route.utterances:
                    all_utterances.append(u)
                    all_route_names.append(route.name)

            embeddings = encoder(all_utterances)

            index = LocalIndex()
            index.add(
                embeddings = embeddings,
                routes     = all_route_names,
                utterances = all_utterances,
            )

            self.router = SemanticRouter(
                encoder = encoder,
                routes  = routes,
                index   = index,
            )
            print("[RAG] Router ready!")

        except Exception as e:
            print(f"[RAG] Router init failed: {e}")
            traceback.print_exc()
            self.router = None

    # ── 公共接口 ──────────────────────────────────────────
    def check_route(self, query: str) -> str | None:
        if not self.router:
            return None
        try:
            result = self.router(query)
            print(f"[Router] Route={result.name}  Score={result.similarity_score:.3f}")
            return result.name
        except Exception as e:
            print(f"[Router Error] {e}")
            return None

    def retrieve_context(self, query: str) -> str:
        if not self.retriever:
            return ""
        try:
            docs = self.retriever.invoke(query)
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[Retriever Error] {e}")
            return ""

    @property
    def is_ready(self) -> bool:
        return self.router is not None or self.retriever is not None


# ============================================================
# 硬件控制
# ============================================================
class HardwareController:

    def __init__(self):
        try:
            GPIO.setmode(GPIO.BOARD)
        except:
            pass
        self.lcd = WhisPlayBoard()
        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 20
            )
        except:
            self.font = ImageFont.load_default()
        self.lcd.set_backlight(100)

    def _rgb565(self, img) -> bytes:
        img  = img.convert('RGB')
        data = np.array(img)
        r = (data[:, :, 0] >> 3).astype(np.uint16)
        g = (data[:, :, 1] >> 2).astype(np.uint16)
        b = (data[:, :, 2] >> 3).astype(np.uint16)
        return ((r << 11) | (g << 5) | b).byteswap().tobytes()

    def _flush(self, img):
        self.lcd.draw_image(
            0, 0, self.lcd.LCD_WIDTH, self.lcd.LCD_HEIGHT,
            self._rgb565(img)
        )

    def _canvas(self) -> tuple:
        img  = Image.new("RGB", (self.lcd.LCD_WIDTH, self.lcd.LCD_HEIGHT), "black")
        draw = ImageDraw.Draw(img)
        return img, draw

    def show_idle(self, mode: str = 'chat'):
        """待机界面，显示当前模式"""
        if mode == 'rag':
            self.show_centered("[ RAG Mode ]\nHold: Talk\nDbl: Clear")
        else:
            self.show_centered("[ Chat Mode ]\nHold: Talk\nDbl: Clear")

    def show_centered(self, text: str, color: str = "white"):
        img, draw = self._canvas()
        y = 20
        for line in text.split('\n'):
            bbox = draw.textbbox((0, 0), line, font=self.font)
            x    = (self.lcd.LCD_WIDTH - (bbox[2] - bbox[0])) // 2
            draw.text((x, y), line, fill=color, font=self.font)
            y += 38
        self._flush(img)

    def show_response(self, text: str):
        img, draw = self._canvas()
        clean = text.replace("*", "").replace("#", "")
        lines = []
        for para in clean.split('\n'):
            if not para:
                lines.append("")
            else:
                lines.extend(textwrap.wrap(para, width=25))
        show = lines[-8:] if len(lines) > 8 else lines
        y = 16
        for line in show:
            draw.text((10, y), line, font=self.font, fill="white")
            y += 30
        self._flush(img)

    def is_pressed(self) -> bool:
        return self.lcd.button_pressed()

    def stop_audio(self):
        try:
            requests.post(TTS_STOP_URL, timeout=1)
        except:
            pass


# ============================================================
# 系统优化
# ============================================================
def optimize_priority():
    try:
        pids = subprocess.check_output(
            ["pgrep", "-f", "ollama_llama"]
        ).decode().strip().split()
        for pid in pids:
            if pid:
                subprocess.run(
                    f"sudo renice -n 5 -p {pid}",
                    shell=True, stderr=subprocess.DEVNULL
                )
        print("[System] Ollama reniced +5")
    except:
        pass


# ============================================================
# TTS 辅助
# ============================================================
def tts_send(text: str, log_first: bool = False) -> bool:
    safe = text.replace('*', '').replace('#', '').replace('\n', ' ').strip()
    if len(safe) < MIN_SENT_LEN:
        return False
    try:
        r = requests.post(
            TTS_SPEAK_URL,
            json    = {"text": safe, "language": "en"},
            timeout = TTS_TIMEOUT,
        )
        if r.status_code == 200 and log_first:
            print(f"[TTS] First sentence: \"{safe[:40]}\"")
            time.sleep(0.15)
        return r.status_code == 200
    except requests.Timeout:
        print("[TTS] Send timeout")
        return False
    except Exception as e:
        print(f"[TTS] Send error: {e}")
        return False


# ============================================================
# RAG prompt 构建
# ============================================================
def _build_prompt(hw: HardwareController, user_text: str) -> str:
    """
    RAG 模式：路由 → 检索 → 拼接 prompt
    Chat 模式 / 路由失败 / out_of_domain：直接返回原始 user_text
    """
    global current_mode, rag_engine

    if current_mode != 'rag' or rag_engine is None or not rag_engine.is_ready:
        return user_text

    # 路由
    hw.show_centered("Routing...")
    route = rag_engine.check_route(user_text)
    print(f"[RAG] Route: {route}")

    if not route or route == "out_of_domain":
        hw.show_centered("Direct LLM\n(no match)")
        time.sleep(0.4)
        return user_text

    # 检索
    hw.show_centered(f"Searching:\n{route[:18]}...")
    context = rag_engine.retrieve_context(user_text)

    if not context:
        print("[RAG] No context found, using direct LLM")
        return user_text

    print(f"[RAG] Context: {len(context)} chars")
    return (
        f"Reference material:\n{context}\n\n"
        f"Question: {user_text}\n"
        f"Answer the question using the reference material above. "
        f"Be concise and conversational."
    )


# ============================================================
# 核心对话流程
# ============================================================
def respond(hw: HardwareController, user_text: str):
    """
    流式调用 LLM → 实时切句推送 TTS → 节流刷新 LCD
    RAG 模式下先执行路由+检索，再调 LLM。
    """
    global chat_history

    stop_event.clear()
    hw.stop_audio()

    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]

    # RAG：构建最终 prompt
    final_prompt = _build_prompt(hw, user_text)

    # RAG 模式：不带 chat history，每次独立检索回答
    # Chat 模式：带完整历史，保持对话连贯性
    if current_mode == 'rag':
        messages_to_send = [{'role': 'user', 'content': final_prompt}]
    else:
        chat_history.append({'role': 'user', 'content': user_text})
        messages_to_send = chat_history[:-1] + [
            {'role': 'user', 'content': final_prompt}
        ]

    # 计时
    t_start      = time.time()
    t_llm        = 0.0

    full_resp    = ""
    cur_sent     = ""
    pending_buf  = ""
    is_first_tok = True
    is_first_tts = True
    last_screen  = time.time()

    try:
        print("\n[LLM] ", end='', flush=True)
        t_llm = time.time()

        stream = ollama.chat(
            model      = MODEL_NAME,
            messages   = messages_to_send,
            stream     = True,
            options    = RAG_LLM_OPTIONS if current_mode == 'rag' else LLM_OPTIONS,
            keep_alive = -1,   # 对话结束后保留在内存
        )

        for chunk in stream:
            if stop_event.is_set():
                print("\n[LLM] Interrupted")
                break

            token = chunk['message']['content']

            if is_first_tok:
                t_ttft       = time.time()
                is_first_tok = False
                print(f"\n[Perf] TTFT: {t_ttft - t_llm:.2f}s")

            print(token, end='', flush=True)
            full_resp += token
            cur_sent  += token

            now = time.time()
            if now - last_screen >= SCREEN_INTERVAL:
                hw.show_response(full_resp)
                last_screen = now

            slen       = len(cur_sent)
            should_cut = (
                any(p in token for p in STRONG_PUNCT)
                or (any(p in token for p in WEAK_PUNCT) and slen > 15)
                or slen >= FORCE_SPLIT_LEN
            )

            if should_cut:
                to_play  = cur_sent.strip()
                cur_sent = ""

                if len(to_play) < MIN_SENT_LEN:
                    pending_buf += (" " + to_play if pending_buf else to_play)
                    continue

                if pending_buf:
                    to_play     = pending_buf + " " + to_play
                    pending_buf = ""

                if is_first_tts and len(to_play) < FIRST_SENT_MIN:
                    pending_buf = to_play
                    continue

                if is_first_tts:
                    t_first_tts  = time.time()
                    is_first_tts = False
                    print(f"\n[Perf] First TTS: {t_first_tts - t_llm:.2f}s")
                    tts_send(to_play, log_first=True)
                else:
                    tts_send(to_play)

        t_end = time.time()
        dur   = t_end - t_llm
        chars = len(full_resp)
        print(f"\n[Perf] LLM: {dur:.2f}s | {chars}chars | {chars/dur:.1f}c/s" if dur > 0 else "")
        print(f"[Perf] Pipeline: {t_end - t_start:.2f}s")

        tail = " ".join(filter(None, [pending_buf, cur_sent.strip()]))
        if tail:
            tts_send(tail)

    except Exception as e:
        print(f"\n[LLM Error] {e}")
        traceback.print_exc()

    if not full_resp.strip():
        full_resp = "Sorry, I didn't catch that. Could you try again?"
        tts_send(full_resp)

    # RAG 模式：不存历史（每次独立，避免 prefill 累积）
    # Chat 模式：存历史（保持对话连贯）
    if current_mode != 'rag':
        chat_history.append({'role': 'assistant', 'content': full_resp})

    hw.show_response(full_resp)


# ============================================================
# 服务健康检查
# ============================================================
def wait_for_service(name: str, url: str,
                     retries: int = 30, interval: float = 2.0) -> bool:
    for i in range(retries):
        try:
            r = requests.get(url, timeout=SVC_TIMEOUT)
            if r.status_code == 200:
                print(f"[INIT] {name} ready!")
                return True
        except:
            pass
        print(f"[INIT] Waiting for {name}... ({i+1}/{retries})")
        time.sleep(interval)
    return False


# ============================================================
# 主程序
# ============================================================
def main():
    global chat_history, current_mode, rag_engine

    chat_history = []

    # 1. 硬件初始化
    optimize_priority()
    hw = HardwareController()
    hw.show_centered("Starting...")

    # 2. ASR
    hw.show_centered("Checking ASR...")
    if not wait_for_service("ASR", "http://localhost:8000/docs"):
        hw.show_centered("ASR Failed!\nCheck service")
        time.sleep(5)
        return

    # 3. TTS
    hw.show_centered("Checking TTS...")
    if not wait_for_service("TTS", TTS_HEALTH_URL):
        hw.show_centered("TTS Failed!\nCheck service")
        time.sleep(5)
        return

    # 4. RAG 初始化（先完成，避免之后挤出 LLM）
    print("[INIT] Initializing RAG Engine...")
    hw.show_centered("Loading RAG\nEngine...")
    try:
        rag_engine = RAGEngine()
        if rag_engine.is_ready:
            print("[INIT] RAG Engine ready!")
        else:
            print("[INIT] RAG not ready, RAG mode will fallback to Chat")
            hw.show_centered("RAG failed\nChat only")
            time.sleep(1)
    except Exception as e:
        print(f"[INIT] RAG init failed: {e}")
        traceback.print_exc()
        rag_engine = None

    # 5. LLM 预热（RAG 已完成释放 nomic，LLM 可以稳定留在内存）
    print("[INIT] Pre-warming LLM...")
    hw.show_centered("Loading LLM...")
    t_warm = time.time()
    try:
        ollama.chat(
            model      = MODEL_NAME,
            messages   = [{'role': 'user', 'content': 'hi'}],
            options    = {'num_thread': 4, 'num_predict': 1},
            keep_alive = -1,   # 永久保留在内存，不自动卸载
        )
        print(f"[INIT] LLM ready! ({time.time()-t_warm:.1f}s)")
    except Exception as e:
        print(f"[INIT] LLM warmup failed: {e}")
        hw.show_centered("LLM Failed!\nCheck Ollama")
        time.sleep(3)

    # 6. 就绪
    hw.show_idle(current_mode)
    print(f"[INIT] System ready! Mode={current_mode}\n")

    # 7. 主循环
    while True:
        try:
            if not hw.is_pressed():
                time.sleep(0.05)
                continue

            press_start = time.time()
            is_long     = False

            while hw.is_pressed():
                held = time.time() - press_start

                if held >= LONG_PRESS_SEC and not is_long:
                    # 长按：录音
                    is_long = True
                    stop_event.set()
                    hw.stop_audio()
                    hw.show_centered("Listening...")

                    audio_chunks = []
                    with sd.InputStream(
                        samplerate = SAMPLE_RATE,
                        channels   = 2,
                        device     = INPUT_DEVICE_ID
                    ) as mic:
                        while hw.is_pressed():
                            chunk, _ = mic.read(1024)
                            audio_chunks.append(chunk[:, 0])

                time.sleep(0.01)

            if is_long:
                hw.show_centered("Thinking...")
                pipeline_t = time.time()

                rec      = np.concatenate(audio_chunks)
                duration = len(rec) / SAMPLE_RATE
                print(f"[Audio] Recorded {duration:.2f}s")

                if duration < MIN_RECORD_SEC:
                    print("[Audio] Too short, ignored")
                    hw.show_idle(current_mode)
                    continue

                wav.write(WAV_PATH, SAMPLE_RATE, (rec * 32767).astype(np.int16))

                # ASR
                u_text = ""
                t_asr  = time.time()
                try:
                    with open(WAV_PATH, 'rb') as f:
                        r = requests.post(
                            ASR_URL,
                            files   = {'file': ('req.wav', f, 'audio/wav')},
                            timeout = ASR_TIMEOUT,
                        )
                    print(f"[Perf] ASR: {time.time()-t_asr:.2f}s")

                    if r.status_code == 200:
                        data   = r.json()
                        u_text = data.get("text", "").strip()
                        lang   = data.get("language", "?")
                        print(f"[ASR] \"{u_text}\" (lang:{lang})")
                    else:
                        hw.show_centered("ASR Error")
                        time.sleep(1)

                except requests.Timeout:
                    hw.show_centered("ASR Timeout")
                    time.sleep(1)
                except Exception as e:
                    print(f"[ASR] {e}")
                    hw.show_centered("ASR Offline")
                    time.sleep(1)

                # LLM + TTS（子线程，主线程监听打断）
                if u_text:
                    respond_thread = threading.Thread(
                        target = respond,
                        args   = (hw, u_text),
                        daemon = True,
                        name   = "RespondThread",
                    )
                    respond_thread.start()

                    while respond_thread.is_alive():
                        if hw.is_pressed():
                            press_t = time.time()
                            interrupted = False
                            while hw.is_pressed():
                                if time.time() - press_t >= LONG_PRESS_SEC:
                                    print("\n[System] Interrupted!")
                                    stop_event.set()
                                    hw.stop_audio()
                                    hw.show_centered("Interrupted")
                                    respond_thread.join(timeout=3)
                                    time.sleep(0.5)
                                    hw.show_idle(current_mode)
                                    interrupted = True
                                    break
                                time.sleep(0.01)
                            
                            if interrupted:
                                break
                            # 短按：不 break，继续监听线程
                        
                        time.sleep(0.05)

                    if respond_thread.is_alive():
                        respond_thread.join(timeout=5)

                    print(f"[Perf] Full Pipeline: {time.time()-pipeline_t:.2f}s")
                else:
                    hw.show_idle(current_mode)

            else:
                # 短按：判断单击 or 双击
                t_click = time.time()
                double  = False
                while time.time() - t_click < 0.35:
                    if hw.is_pressed():
                        double = True
                        while hw.is_pressed():
                            time.sleep(0.01)
                        break
                    time.sleep(0.01)

                if double:
                    # 双击：清空历史
                    stop_event.set()
                    hw.stop_audio()
                    chat_history = []
                    hw.show_centered("Memory\nCleared!")
                    print("[System] Chat history cleared")
                    time.sleep(0.8)
                    hw.show_idle(current_mode)

                else:
                    # 单击：切换 Chat / RAG 模式
                    current_mode = 'rag' if current_mode == 'chat' else 'chat'
                    label = "RAG Mode" if current_mode == 'rag' else "Chat Mode"
                    hw.show_centered(f"Switched to\n{label}")
                    print(f"[System] Mode → {current_mode}")
                    time.sleep(0.8)
                    hw.show_idle(current_mode)

        except KeyboardInterrupt:
            print("\n[System] Shutting down...")
            break
        except Exception as e:
            print(f"[Main Loop] Error: {e}")
            traceback.print_exc()
            time.sleep(1)

    # 清理
    stop_event.set()
    try:
        requests.post(TTS_STOP_URL, timeout=1)
    except:
        pass
    GPIO.cleanup()
    print("[System] Goodbye!")


if __name__ == "__main__":
    main()