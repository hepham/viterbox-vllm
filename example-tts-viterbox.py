"""
Example: Viterbox Vietnamese TTS on vLLM
"""
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS


if __name__ == "__main__":
    print("Loading Viterbox model on vLLM...")
    model = ChatterboxTTS.from_pretrained_viterbox(
        max_model_len=1000,
        max_batch_size=5,
        # Disable CUDA graphs for one-off generation
        compile=False,
    )
    print("Model loaded!")

    # Vietnamese text
    prompts = [
        "Xin chào! Đây là mô hình chuyển văn bản thành giọng nói tiếng Việt.",
        "Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố.",
    ]

    # Use default voice or specify audio_prompt_path for voice cloning
    audio_prompt_path = None  # or "path/to/reference.wav"

    print(f"Generating {len(prompts)} prompts...")
    audios = model.generate(
        prompts,
        audio_prompt_path=audio_prompt_path,
        language_id="vi",
        exaggeration=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
    )

    for i, audio in enumerate(audios):
        output_path = f"viterbox_vllm_output_{i}.wav"
        ta.save(output_path, audio, model.sr)
        print(f"Saved: {output_path}")

    print("\nDone!")
