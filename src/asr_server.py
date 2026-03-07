import uvicorn
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import os
import shutil
import time

# --- 配置区域 ---
MODEL_PATH = "/home/user/AITutor/models/whisper-base"
# 树莓派优化：使用 int8 量化，限制线程数防止卡死 UI
COMPUTE_TYPE = "int8" 
CPU_THREADS = 3 
PORT = 8000

app = FastAPI()
model = None

# --- 生命周期管理 ---
@app.on_event("startup")
def load_model():
    global model
    print(f"[ASR Server] Loading model from {MODEL_PATH}...")
    start_t = time.time()
    # 核心：模型加载只发生在这里，一次性完成
    model = WhisperModel(MODEL_PATH, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
    print(f"[ASR Server] Model loaded in {time.time() - start_t:.2f}s")

# --- API 接口 ---
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 1. 保存上传的临时文件
    temp_filename = f"temp_{int(time.time())}.wav"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        start_t = time.time()
        # 2. 执行推理 (此时模型已在内存，速度极快)
        segments, info = model.transcribe(temp_filename, beam_size=5)
        text = "".join([s.text for s in segments]).strip()
        
        process_time = time.time() - start_t
        print(f"[ASR] Processed in {process_time:.2f}s | Text: {text}")
        
        return {"text": text, "language": info.language, "duration": process_time}
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # 3. 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=PORT)