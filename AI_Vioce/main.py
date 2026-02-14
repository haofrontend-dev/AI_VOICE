# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/OpenBMB/VoxCPM.git
# %cd VoxCPM
!pip install -q voxcpm torchcodec==0.9

"""# Test on CLI"""

from huggingface_hub import snapshot_download
snapshot_download("JayLL13/VoxCPM-1.5-VN", local_dir="./pretrained/VoxCPM-1.5-VN")


# =========================
# VoxCPM - Dark UI (VN) - Click-to-Generate
# =========================
import os
import gradio as gr
import soundfile as sf
import torch
from voxcpm import VoxCPM

# ---- Torch perf knobs ----
os.environ["TORCHDYNAMO_DISABLE"] = "1"
try:
    torch._dynamo.disable()
except Exception:
    pass

try:
    import torch._inductor.config as config
    config.triton.cudagraphs = False
except Exception:
    pass

torch.backends.cuda.matmul.allow_tf32 = True

# Set default dtype to float32 to prevent bfloat16 issues on CPU
torch.set_default_dtype(torch.float32)

# ---- Paths ----
INPUT_AUDIO_PATH = "input.wav"
OUTPUT_AUDIO_PATH = "output.wav"

sample_prompt_text = (
    "Làm content mà ngại thu âm thì phải làm sao? "
    "Hôm nay mình demo cho anh em cách mình dùng."
)

# ---- Load model once ----
print("Loading VoxCPM model...")
model = VoxCPM.from_pretrained("JayLL13/VoxCPM-1.5-VN")
# The explicit .to(torch.float32) call here is technically redundant if dtype is passed correctly, but harmless
# model.to(torch.float32) # Removed this line as it causes an AttributeError
print("Model loaded!")

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def luu_ghi_am(audio_np):
    """audio_np: (sr, np.ndarray) with gr.Audio(type='numpy')"""
    if audio_np is None:
        return (
            gr.update(value="❌ Chưa có âm thanh. Hãy ghi âm hoặc upload file."),
            gr.update(value=False),
        )

    sr, data = audio_np
    sf.write(INPUT_AUDIO_PATH, data, sr)
    return (
        gr.update(value=f"✅ Đã lưu: {INPUT_AUDIO_PATH} (sr={sr})"),
        gr.update(value=True),
    )

@torch.inference_mode()
def tao_giong_noi(
    van_ban_muc_tieu: str,
    van_ban_mau: str,
    cfg_value: float,
    so_buoc: int,
    chuan_hoa: bool,
    khu_nhieu: bool,
    tu_thu_lai: bool,
):
    van_ban_muc_tieu = (van_ban_muc_tieu or "").strip()
    van_ban_mau = (van_ban_mau or "").strip()

    if van_ban_muc_tieu == "":
        return None, "⏳ **Nhập văn bản mục tiêu** để bắt đầu."

    if not os.path.exists(INPUT_AUDIO_PATH):
        return None, "🎤 **Chưa có giọng mẫu**. Hãy ghi âm và bấm **Lưu ghi âm** trước."

    # Clamp để không crash nếu user gõ trực tiếp ngoài range
    so_buoc = _clamp(int(so_buoc), 4, 30)
    cfg_value = _clamp(float(cfg_value), 1.0, 3.0)

    # VoxCPM requirement: có prompt_wav thì prompt_text không được None/rỗng
    if van_ban_mau == "":
        return None, "❌ Có **giọng mẫu** thì **Văn bản giọng mẫu** không được để trống (phải khớp nội dung bạn đọc)."

    status = f"🚀 Đang tạo giọng... (CFG={cfg_value:.1f}, Steps={so_buoc})"

    wav = model.generate(
        text=van_ban_muc_tieu,
        prompt_wav_path=INPUT_AUDIO_PATH,
        prompt_text=van_ban_mau,
        cfg_value=cfg_value,
        inference_timesteps=so_buoc,
        normalize=bool(chuan_hoa),
        denoise=bool(khu_nhieu),
        retry_badcase=bool(tu_thu_lai),
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )

    sf.write(OUTPUT_AUDIO_PATH, wav, model.tts_model.sample_rate)
    return OUTPUT_AUDIO_PATH, f"✅ **Xong!** · CFG={cfg_value:.1f} · Steps={so_buoc}"

# =========================
# UI Theme + CSS (Dark, Card)
# =========================
dark_css = """
:root { color-scheme: dark; }

.gradio-container {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(59,130,246,0.22), transparent 55%),
              radial-gradient(900px 500px at 80% 10%, rgba(99,102,241,0.18), transparent 55%),
              #070b14 !important;
}

#app_title h1 { font-size: 1.35rem !important; margin-bottom: 0.2rem !important; }
#sub_title { opacity: 0.9; margin-bottom: 0.8rem; }

.card {
  background: rgba(17, 24, 39, 0.86) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 18px !important;
  padding: 14px !important;
}

textarea, input[type="text"] {
  background: rgba(2, 6, 23, 0.85) !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
}

button { border-radius: 14px !important; }

.hr { height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0 12px 0; }
.small_note { font-size: 0.88rem; opacity: 0.9; }
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
).set(
    body_background_fill="#070b14",
    block_background_fill="rgba(17, 24, 39, 0.86)",
)

# =========================
# Build UI (CLICK ONLY)
# =========================
with gr.Blocks(theme=theme, css=dark_css, title="VoxCPM Voice Cloning (VN)") as demo:
    gr.Markdown("# 🎤 VoxCPM Voice Cloning (VN • Dark)", elem_id="app_title")
    gr.Markdown(
        "Bấm **Lưu ghi âm** để lưu giọng mẫu, sau đó bấm **Tạo giọng** để generate. "
        "Gợi ý: **CFG 1.8–2.2**, **Steps 18–22**.",
        elem_id="sub_title",
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("## 1) Giọng mẫu (bắt buộc)")
            gr.Markdown("💡 *Ghi âm 10–25s, nói tự nhiên, tránh nhạc nền/echo.*", elem_classes=["small_note"])

            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Âm thanh giọng mẫu (Mic / Tải lên)",
            )
            btn_save = gr.Button("🎤 Lưu ghi âm", variant="primary")
            save_status = gr.Textbox(label="Trạng thái", interactive=False)
            has_audio_flag = gr.Checkbox(label="Đã có giọng mẫu", value=False, interactive=False)

            gr.HTML('<div class="hr"></div>')

            gr.Markdown("## 2) Văn bản giọng mẫu")
            prompt_text = gr.Textbox(
                value=sample_prompt_text,
                lines=4,
                label="Văn bản đúng như bạn đã đọc (khớp 100%)",
            )

        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("## 3) Văn bản mục tiêu")
            target_text = gr.Textbox(
                value="Xin chào các bạn. Tôi tên là Anh Đức. Tôi rất là xấu trai. Các bạn nói là đúng đi",
                lines=4,
                label="Văn bản muốn chuyển sang giọng đã clone",
            )

            with gr.Tabs():
                with gr.Tab("Cơ bản"):
                    with gr.Row():
                        cfg = gr.Slider(
                            1.0, 3.0, value=2.0, step=0.1,
                            label="CFG (mức bám giọng)",
                            info="Tăng để giống giọng mẫu hơn; giảm để tự nhiên hơn"
                        )
                        steps = gr.Slider(
                            4, 30, value=20, step=1,
                            label="Số bước suy luận",
                            info="Nhiều hơn thường mượt hơn nhưng chậm hơn"
                        )
                with gr.Tab("Nâng cao"):
                    with gr.Row():
                        chuan_hoa = gr.Checkbox(value=False, label="Chuẩn hoá văn bản")
                        khu_nhieu = gr.Checkbox(value=False, label="Khử nhiễu giọng mẫu")
                        tu_thu_lai = gr.Checkbox(value=True, label="Tự thử lại khi lỗi")

            btn_generate = gr.Button("⚡ Tạo giọng", variant="primary")
            out_audio = gr.Audio(label="Âm thanh đầu ra", type="filepath")
            status_md = gr.Markdown("⏳ Chưa chạy.")

    # Wiring: ONLY CLICK
    btn_save.click(fn=luu_ghi_am, inputs=audio_in, outputs=[save_status, has_audio_flag])

    btn_generate.click(
        fn=tao_giong_noi,
        inputs=[target_text, prompt_text, cfg, steps, chuan_hoa, khu_nhieu, tu_thu_lai],
        outputs=[out_audio, status_md],
        show_progress=True,
    )

demo.queue(max_size=10, default_concurrency_limit=1).launch(debug=True)