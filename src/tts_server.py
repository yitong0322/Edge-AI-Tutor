#!/usr/bin/env python3
"""
TTS Service - 基于 Piper 的高性能 TTS 服务
端口: 8001
"""

import os
import json
import subprocess
import threading
import queue
import time
import signal
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ============ 配置 ============
PROJECT_DIR    = "/home/user/AITutor"
PIPER_EXE      = os.path.join(PROJECT_DIR, "bin/piper")
PIPER_MODEL_EN = os.path.join(PROJECT_DIR, "models/en_US-lessac-low.onnx")
OUTPUT_DEVICE  = "default"
APLAY_RATE     = 16000

# ============ 全局变量 ============
app = FastAPI(title="TTS Service", version="2.0")

piper_proc  = None
proc_lock   = threading.Lock()
tts_queue   = queue.Queue(maxsize=20)
worker_stop = threading.Event()


# ============ 数据模型 ============
class TTSRequest(BaseModel):
    text: str
    language: str = "en"

class TTSResponse(BaseModel):
    status: str
    message: str = ""


# ============ Piper 进程管理 ============
def _kill_piper():
    """强制杀掉当前 Piper 进程及其 aplay 子进程"""
    global piper_proc
    with proc_lock:
        if piper_proc is not None:
            try:
                # 杀整个进程组（包含 shell 和 aplay）
                os.killpg(os.getpgid(piper_proc.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                piper_proc.wait(timeout=1)
            except Exception:
                pass
            piper_proc = None
    subprocess.run("pkill -9 aplay", shell=True, stderr=subprocess.DEVNULL)


def _start_piper() -> subprocess.Popen | None:
    """启动一个新的 Piper 进程，返回进程对象"""
    global piper_proc
    cmd = (
        f'nice -n -10 {PIPER_EXE} --model {PIPER_MODEL_EN} '
        f'--json-input --output_raw --length_scale 0.85 | '
        f'nice -n -5 aplay -D {OUTPUT_DEVICE} -r {APLAY_RATE} '
        f'-f S16_LE -t raw -c 1 --buffer-time=50000'
        # buffer-time=50000us(50ms)，给 Piper 更多缓冲，避免首句截断
    )
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True   # 创建独立进程组，方便整组杀掉
        )
        print(f"[Piper] Started PID={proc.pid}")
        # 给 Piper 短暂预热时间，避免第一条文本被丢弃
        time.sleep(0.15)
        return proc
    except Exception as e:
        print(f"[Piper] Start failed: {e}")
        return None


def get_piper() -> subprocess.Popen | None:
    """获取可用的 Piper 进程，进程死了自动重启"""
    global piper_proc
    with proc_lock:
        if piper_proc is None or piper_proc.poll() is not None:
            print("[Piper] Process not running, restarting...")
            piper_proc = _start_piper()
        return piper_proc


def send_to_piper(text: str) -> bool:
    """
    把一段文本写入 Piper stdin。
    写失败时自动重启 Piper 并重试一次。
    """
    global piper_proc
    payload = json.dumps({"text": text}) + "\n"
    encoded = payload.encode('utf-8')

    for attempt in range(2):
        proc = get_piper()
        if proc is None:
            print("[Piper] No process available")
            return False
        try:
            proc.stdin.write(encoded)
            proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError) as e:
            print(f"[Piper] Pipe error (attempt {attempt+1}): {e}")
            # 管道断了，标记为需要重启
            with proc_lock:
                piper_proc = None
            if attempt == 0:
                time.sleep(0.1)   # 短暂等待后重试
    return False


# ============ TTS 工作线程 ============
def tts_worker():
    print("[TTS Worker] Started")
    while not worker_stop.is_set():
        try:
            task = tts_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if task is None:   # 关闭信号
            tts_queue.task_done()
            break

        text = task
        safe = text.replace('*', '').replace('#', '').replace('\n', ' ').strip()

        if safe:
            ok = send_to_piper(safe)
            if ok:
                print(f"[TTS Worker] Sent: {safe[:60]}{'...' if len(safe)>60 else ''}")
            else:
                print(f"[TTS Worker] Failed to send: {safe[:40]}")

        tts_queue.task_done()

    print("[TTS Worker] Stopped")


# ============ FastAPI 生命周期 ============
@app.on_event("startup")
def startup():
    global piper_proc
    print("[TTS Service] Starting up...")
    os.environ['ORT_LOGGING_LEVEL'] = '3'

    # 预启动 Piper（这样第一句话不需要等待 Piper 初始化）
    piper_proc = _start_piper()
    if piper_proc:
        print("[TTS Service] Piper pre-warmed!")
    else:
        print("[TTS Service] WARNING: Piper pre-warm failed, will retry on first request")

    # 启动工作线程
    t = threading.Thread(target=tts_worker, daemon=True, name="TTS-Worker")
    t.start()

    print("[TTS Service] Ready on port 8001")


@app.on_event("shutdown")
def shutdown():
    print("[TTS Service] Shutting down...")
    worker_stop.set()
    tts_queue.put(None)
    _kill_piper()
    print("[TTS Service] Stopped")


# ============ API 端点 ============
@app.post("/speak", response_model=TTSResponse)
async def speak(request: TTSRequest):
    """把文本加入 TTS 队列"""
    if tts_queue.full():
        return TTSResponse(status="error", message="Queue full")

    # 英文 only，忽略 language 字段
    tts_queue.put(request.text)
    return TTSResponse(status="success", message="Queued")


@app.post("/stop")
async def stop_audio():
    """
    立即停止播放：
    1. 清空待播队列
    2. 杀掉 Piper + aplay
    3. 重启 Piper（为下一轮做好准备）
    """
    global piper_proc

    # 1. 清空队列
    cleared = 0
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
            tts_queue.task_done()
            cleared += 1
        except Exception:
            break

    # 2. 杀掉当前播放
    _kill_piper()

    # 3. 重启 Piper（异步，避免阻塞 HTTP 响应）
    def _restart():
        global piper_proc
        time.sleep(0.1)
        with proc_lock:
            piper_proc = _start_piper()
        print("[Piper] Restarted after stop")

    threading.Thread(target=_restart, daemon=True).start()

    return TTSResponse(status="success", message=f"Stopped, cleared {cleared} items")


@app.get("/health")
async def health():
    with proc_lock:
        alive = piper_proc is not None and piper_proc.poll() is None

    return {
        "status":     "healthy",
        "piper_en":   "running" if alive else "stopped",
        "queue_size": tts_queue.qsize(),
    }


# ============ 启动 ============
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")