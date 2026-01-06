"""
Compare audio quality: Viterbox Original vs Viterbox+vLLM
Using same text and settings for fair comparison
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time
import torch
import torchaudio as ta

TEXT = """Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố, Buổi sáng, thành phố thức dậy cùng mùi hoa sữa thoang thoảng và tiếng rao quen thuộc, Hồ Gươm lặng yên soi bóng những hàng cây xanh, còn những con phố cổ lưu giữ nhịp sống chậm rãi, bình yên, Hà Nội không ồn ào nhưng đủ sâu lắng để khiến ai từng ghé qua đều nhớ mãi"""

AUDIO_PROMPT = "/mnt/d/Project/Python/chatterbox/mau.wav"

# Common generation settings
GEN_SETTINGS = {
    "exaggeration": 0.5,
    "temperature": 0.8,
    "repetition_penalty": 2.0,
}


def benchmark_vllm():
    """Benchmark Viterbox on vLLM"""
    from chatterbox_vllm.tts import ChatterboxTTS
    
    print("\n" + "="*60)
    print("VITERBOX + vLLM")
    print("="*60)
    
    model = ChatterboxTTS.from_pretrained_viterbox(
        max_model_len=1000,
        max_batch_size=5,
        compile=False,
    )
    
    # Warmup
    print("Warmup...")
    _ = model.generate("Xin chào", audio_prompt_path=AUDIO_PROMPT, language_id="vi")
    torch.cuda.synchronize()
    
    # Generate
    print("Generating...")
    torch.cuda.synchronize()
    start = time.time()
    audios = model.generate(
        TEXT, 
        audio_prompt_path=AUDIO_PROMPT, 
        language_id="vi",
        **GEN_SETTINGS
    )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    audio = audios[0]
    audio_duration = audio.shape[1] / model.sr
    
    # Save
    output_path = "/mnt/d/Project/Python/chatterbox/compare_vllm.wav"
    ta.save(output_path, audio, model.sr)
    print(f"Saved: {output_path}")
    
    model.shutdown()
    torch.cuda.empty_cache()
    
    return elapsed, audio_duration


def benchmark_original():
    """Benchmark original Viterbox"""
    import sys
    sys.path.insert(0, "/mnt/d/Project/Python/chatterbox")
    from example_viterbox import ViterboxTTS, punc_norm
    
    print("\n" + "="*60)
    print("VITERBOX ORIGINAL (Chatterbox)")
    print("="*60)
    
    device = "cuda"
    model = ViterboxTTS.from_pretrained(device=device)
    
    # Warmup
    print("Warmup...")
    _ = model.generate("Xin chào", language_id="vi", audio_prompt_path=AUDIO_PROMPT)
    torch.cuda.synchronize()
    
    # Generate
    print("Generating...")
    torch.cuda.synchronize()
    start = time.time()
    wav = model.generate(
        TEXT, 
        language_id="vi", 
        audio_prompt_path=AUDIO_PROMPT,
        exaggeration=GEN_SETTINGS["exaggeration"],
        temperature=GEN_SETTINGS["temperature"],
        repetition_penalty=GEN_SETTINGS["repetition_penalty"],
    )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    audio_duration = wav.shape[1] / model.sr
    
    # Save
    output_path = "/mnt/d/Project/Python/chatterbox/compare_original.wav"
    ta.save(output_path, wav.cpu(), model.sr)
    print(f"Saved: {output_path}")
    
    del model
    torch.cuda.empty_cache()
    
    return elapsed, audio_duration


if __name__ == "__main__":
    print("="*60)
    print("QUALITY COMPARISON: Viterbox Original vs vLLM")
    print("="*60)
    print(f"Text: {TEXT[:60]}...")
    print(f"Audio prompt: {AUDIO_PROMPT}")
    
    # Run vLLM first (to avoid memory issues)
    vllm_time, vllm_dur = benchmark_vllm()
    
    # Clear GPU
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Run original
    orig_time, orig_dur = benchmark_original()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Implementation':<25} {'Time':>10} {'Audio':>10} {'Speed':>10}")
    print("-"*60)
    print(f"{'Viterbox + vLLM':<25} {vllm_time:>9.2f}s {vllm_dur:>9.2f}s {vllm_dur/vllm_time:>9.2f}x")
    print(f"{'Viterbox Original':<25} {orig_time:>9.2f}s {orig_dur:>9.2f}s {orig_dur/orig_time:>9.2f}x")
    print("-"*60)
    speedup = orig_time / vllm_time
    print(f"vLLM is {speedup:.2f}x faster")
    print()
    print("Output files for quality comparison:")
    print("  - compare_vllm.wav")
    print("  - compare_original.wav")
    print("="*60)
