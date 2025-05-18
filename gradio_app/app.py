# gradio_app/app.py

import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile

def voice_chat(audio):
    if audio is None:
        return None

    sr, audio_data = audio

    # Simpan audio input sebagai file WAV sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    try:
        # Kirim file ke backend FastAPI
        with open(audio_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            response = requests.post("http://localhost:8000/voice-chat", files=files)

        # Tangani respons
        if response.status_code == 200:
            # Simpan respons audio sebagai file WAV sementara
            output_audio_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
            with open(output_audio_path, "wb") as f:
                f.write(response.content)
            return output_audio_path
        else:
            print(f"[ERROR] Backend error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        return None

# UI Gradio
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎙️ Voice Chatbot")
        gr.Markdown("Berbicara langsung ke mikrofon dan dapatkan jawaban suara dari asisten AI.")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources="microphone", type="numpy", label="🎤 Rekam Pertanyaan Anda")
                submit_btn = gr.Button("🔁 Submit")
            with gr.Column():
                audio_output = gr.Audio(type="filepath", label="🔊 Balasan dari Asisten")

        submit_btn.click(
            fn=voice_chat,
            inputs=audio_input,
            outputs=audio_output
        )

    demo.launch()

if __name__ == "__main__":
    main()
