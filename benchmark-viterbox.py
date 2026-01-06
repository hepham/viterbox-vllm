"""
Benchmark: Viterbox on vLLM vs Original Chatterbox implementation
"""
import os
# Disable vLLM multiprocessing to avoid tokenizer registry issues
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time
import torch
import torchaudio as ta

TEXT_LONG = """Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố, Buổi sáng, thành phố thức dậy cùng mùi hoa sữa thoang thoảng và tiếng rao quen thuộc, Hồ Gươm lặng yên soi bóng những hàng cây xanh, còn những con phố cổ lưu giữ nhịp sống chậm rãi, bình yên, Hà Nội không ồn ào nhưng đủ sâu lắng để khiến ai từng ghé qua đều nhớ mãi"""

AUDIO_PROMPT = "/mnt/d/Project/Python/chatterbox/mau.wav"
NUM_RUNS = 3


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
    
    ta.save("benchmark_viterbox_vllm.wav", audios[0], model.sr)
    print(f"Saved: benchmark_viterbox_vllm.wav")
    
    model.shutdown()
    torch.cuda.empty_cache()
    
    return avg_time, audio_duration, rtf


if __name__ == "__main__":
    print("="*60)
    print("BENCHMARK: Viterbox on vLLM")
    print("="*60)
    print(f"Text: {TEXT_LONG[:60]}...")
    print(f"Runs: {NUM_RUNS}")
    
    vllm_time, vllm_dur, vllm_rtf = benchmark_viterbox_vllm()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Average time:     {vllm_time:.2f}s")
    print(f"Audio duration:   {vllm_dur:.2f}s")
    print(f"RTF:              {vllm_rtf:.3f}")
    print(f"Speed:            {1/vllm_rtf:.2f}x realtime")
    print("="*60)
