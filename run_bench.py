"""Run both benchmarks separately"""
import subprocess
import sys

chatterbox_code = '''
import time
import torch
import torchaudio as ta
from example_viterbox import ViterboxTTS

TEXT_LONG = "Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố, Buổi sáng, thành phố thức dậy cùng mùi hoa sữa thoang thoảng và tiếng rao quen thuộc, Hồ Gươm lặng yên soi bóng những hàng cây xanh, còn những con phố cổ lưu giữ nhịp sống chậm rãi, bình yên, Hà Nội không ồn ào nhưng đủ sâu lắng để khiến ai từng ghé qua đều nhớ mãi"
AUDIO_PROMPT = "mau.wav"
NUM_RUNS = 5

device = "cuda"
print(f"[Chatterbox+Viterbox] Loading model on {device}...")
model = ViterboxTTS.from_pretrained(device=device)
print("[Chatterbox+Viterbox] Model loaded!")

print("[Chatterbox+Viterbox] Warmup...")
_ = model.generate("Xin chào!", language_id="vi", audio_prompt_path=AUDIO_PROMPT)

print("[Chatterbox+Viterbox] Benchmarking...")
times = []
for i in range(NUM_RUNS):
    torch.cuda.synchronize()
    start = time.time()
    wav = model.generate(TEXT_LONG, language_id="vi", audio_prompt_path=AUDIO_PROMPT)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}s")

avg_time = sum(times) / len(times)
audio_duration = wav.shape[1] / model.sr
rtf = avg_time / audio_duration
print(f"Average: {avg_time:.2f}s, Audio: {audio_duration:.2f}s, RTF: {rtf:.3f}, Realtime: {1/rtf:.1f}x")
ta.save("output_chatterbox_viterbox.wav", wav.cpu(), model.sr)
'''

viterbox_code = '''
import time
import sys
import torch
import torchaudio as ta

sys.path.insert(0, "temp_viterbox")
from viterbox import Viterbox as ViterboxTTS

TEXT_LONG = "Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố, Buổi sáng, thành phố thức dậy cùng mùi hoa sữa thoang thoảng và tiếng rao quen thuộc, Hồ Gươm lặng yên soi bóng những hàng cây xanh, còn những con phố cổ lưu giữ nhịp sống chậm rãi, bình yên, Hà Nội không ồn ào nhưng đủ sâu lắng để khiến ai từng ghé qua đều nhớ mãi"
AUDIO_PROMPT = "mau.wav"
NUM_RUNS = 5

device = "cuda"
print(f"[Viterbox-TTS] Loading model on {device}...")
model = ViterboxTTS.from_pretrained(device=device)
print("[Viterbox-TTS] Model loaded!")

print("[Viterbox-TTS] Warmup...")
_ = model.generate("Xin chào!", language="vi", audio_prompt=AUDIO_PROMPT, split_sentences=False)

print("[Viterbox-TTS] Benchmarking...")
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
print(f"Average: {avg_time:.2f}s, Audio: {audio_duration:.2f}s, RTF: {rtf:.3f}, Realtime: {1/rtf:.1f}x")
ta.save("output_viterbox_original.wav", wav.cpu(), model.sr)
'''

print("=" * 70)
print("BENCHMARK: Chatterbox+Viterbox vs Viterbox-TTS Original")
print("=" * 70)

print("\n>>> Running Chatterbox+Viterbox <<<")
with open("_temp_cb.py", "w", encoding="utf-8") as f:
    f.write(chatterbox_code)
subprocess.run([sys.executable, "_temp_cb.py"], check=True)

print("\n>>> Running Viterbox-TTS Original <<<")
with open("_temp_vb.py", "w", encoding="utf-8") as f:
    f.write(viterbox_code)
subprocess.run([sys.executable, "_temp_vb.py"], check=True)

import os
os.remove("_temp_cb.py")
os.remove("_temp_vb.py")

print("\n" + "=" * 70)
print("DONE - Check output_chatterbox_viterbox.wav and output_viterbox_original.wav")
print("=" * 70)
