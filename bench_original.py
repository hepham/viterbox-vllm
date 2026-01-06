"""Benchmark Viterbox Original only"""
import time
import torch
import torchaudio as ta
import sys

sys.path.insert(0, "D:/Project/Python/chatterbox/temp_viterbox")
from viterbox import Viterbox as ViterboxTTS

TEXT_LONG = """Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố, Buổi sáng, thành phố thức dậy cùng mùi hoa sữa thoang thoảng và tiếng rao quen thuộc, Hồ Gươm lặng yên soi bóng những hàng cây xanh, còn những con phố cổ lưu giữ nhịp sống chậm rãi, bình yên, Hà Nội không ồn ào nhưng đủ sâu lắng để khiến ai từng ghé qua đều nhớ mãi"""
AUDIO_PROMPT = "D:/Project/Python/chatterbox/mau.wav"
NUM_RUNS = 5

print("="*60)
print("VITERBOX-TTS ORIGINAL")
print("="*60)

device = "cuda"
print(f"Loading model on {device}...")
model = ViterboxTTS.from_pretrained(device=device)
print("Model loaded!")

print("Warmup...")
_ = model.generate("Xin chào", language="vi", audio_prompt=AUDIO_PROMPT, split_sentences=False)
torch.cuda.synchronize()

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

print(f"\nAverage: {avg_time:.2f}s")
print(f"Audio duration: {audio_duration:.2f}s")
print(f"RTF: {rtf:.3f}")
print(f"Speed: {1/rtf:.1f}x realtime")

ta.save("D:/Project/Python/chatterbox/output_viterbox_original.wav", wav.cpu(), model.sr)
print("Saved: output_viterbox_original.wav")
