"""
Benchmark: Viterbox vLLM vs Viterbox Original
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time
import torch
import torchaudio as ta
import gc

TEXT_LONG = """Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố, Buổi sáng, thành phố thức dậy cùng mùi hoa sữa thoang thoảng và tiếng rao quen thuộc, Hồ Gươm lặng yên soi bóng những hàng cây xanh, còn những con phố cổ lưu giữ nhịp sống chậm rãi, bình yên, Hà Nội không ồn ào nhưng đủ sâu lắng để khiến ai từng ghé qua đều nhớ mãi"""

AUDIO_PROMPT = "D:/Project/Python/chatterbox/mau.wav"
NUM_RUNS = 5


def benchmark_viterbox_vllm():
    """Benchmark Viterbox on vLLM"""
    from chatterbox_vllm.tts import ChatterboxTTS
    
    print("\n" + "="*60)
    print("VITERBOX on vLLM")
    print("="*60)
    
    print("Loading model...")
    start_load = time.time()
    model = ChatterboxTTS.from_pretrained_viterbox(
        max_model_len=1000,
        max_batch_size=5,
        compile=False,
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Warmup
    print("Warmup...")
    _ = model.generate("Xin chào", audio_prompt_path=AUDIO_PROMPT, language_id="vi")
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({NUM_RUNS} runs)...")
    times = []
    for i in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.time()
        audios = model.generate(TEXT_LONG, audio_prompt_path=AUDIO_PROMPT, language_id="vi")
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    audio_duration = audios[0].shape[1] / model.sr
    rtf = avg_time / audio_duration
    
    ta.save("output_viterbox_vllm.wav", audios[0], model.sr)
    print(f"Saved: output_viterbox_vllm.wav")
    
    model.shutdown()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_time, audio_duration, rtf


def benchmark_viterbox_original():
    """Benchmark original Viterbox-TTS"""
    import sys
    sys.path.insert(0, "D:/Project/Python/chatterbox/temp_viterbox")
    from viterbox import Viterbox as ViterboxTTS
    
    print("\n" + "="*60)
    print("VITERBOX-TTS ORIGINAL")
    print("="*60)
    
    device = "cuda"
    print(f"Loading model on {device}...")
    start_load = time.time()
    model = ViterboxTTS.from_pretrained(device=device)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Warmup
    print("Warmup...")
    _ = model.generate("Xin chào", language="vi", audio_prompt=AUDIO_PROMPT, split_sentences=False)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({NUM_RUNS} runs)...")
    times = []
    for i in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.time()
        wav = model.generate(TEXT_LONG, language="vi", audio_prompt=AUDIO_PROMPT, split_sentences=False)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    audio_duration = wav.shape[1] / model.sr
    rtf = avg_time / audio_duration
    
    ta.save("output_viterbox_original.wav", wav.cpu(), model.sr)
    print(f"Saved: output_viterbox_original.wav")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_time, audio_duration, rtf


if __name__ == "__main__":
    print("="*70)
    print("BENCHMARK: Viterbox vLLM vs Viterbox Original")
    print("="*70)
    print(f"Text: {TEXT_LONG[:60]}...")
    print(f"Runs: {NUM_RUNS}")
    
    # Run vLLM first
    vllm_time, vllm_dur, vllm_rtf = benchmark_viterbox_vllm()
    
    # Clear GPU completely
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Run original
    orig_time, orig_dur, orig_rtf = benchmark_viterbox_original()
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Implementation':<25} {'Time':<10} {'Audio':<10} {'RTF':<10} {'Speed'}")
    print("-"*70)
    print(f"{'Viterbox vLLM':<25} {vllm_time:.2f}s{'':<5} {vllm_dur:.2f}s{'':<5} {vllm_rtf:.3f}{'':<5} {1/vllm_rtf:.1f}x realtime")
    print(f"{'Viterbox Original':<25} {orig_time:.2f}s{'':<5} {orig_dur:.2f}s{'':<5} {orig_rtf:.3f}{'':<5} {1/orig_rtf:.1f}x realtime")
    print("-"*70)
    
    if vllm_time < orig_time:
        speedup = orig_time / vllm_time
        print(f">>> vLLM is {speedup:.2f}x FASTER than Original")
    else:
        speedup = vllm_time / orig_time
        print(f">>> Original is {speedup:.2f}x FASTER than vLLM")
    print("="*70)
