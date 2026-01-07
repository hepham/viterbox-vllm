"""
Example: High-throughput batch TTS processing

This example demonstrates efficient batch processing of multiple prompts
with voice cloning and custom parameters.
"""
import torchaudio as ta
from chatterbox_vllm.tts import ChatterboxTTS


def main():
    print("Loading model with batch optimization...")
    model = ChatterboxTTS.from_pretrained(
        max_model_len=1000,
        max_batch_size=10,  # Process up to 10 prompts at once
        compile=True,  # Enable CUDA graphs for speed
    )
    print("Model loaded!")

    # Define multiple prompts
    prompts = [
        "Welcome to the Chatterbox TTS demo.",
        "This is the second sentence being processed in parallel.",
        "Batch processing significantly improves throughput.",
        "Each prompt is converted to audio simultaneously.",
        "The final example demonstrates voice cloning capabilities.",
    ]

    # Optional: Use a voice reference for cloning
    voice_reference = None  # or "path/to/reference.wav"

    print(f"\nGenerating {len(prompts)} audio clips...")
    
    # Generate with custom parameters
    audios = model.generate(
        prompts,
        audio_prompt_path=voice_reference,
        exaggeration=0.5,       # Emotion intensity
        cfg_scale=0.5,          # Conditioning strength
        temperature=0.8,        # Sampling randomness
        diffusion_steps=10,     # Audio quality (5-10)
        repetition_penalty=2.0, # Reduce repetition
    )

    # Save outputs
    for i, audio in enumerate(audios):
        output_path = f"batch_output_{i}.wav"
        ta.save(output_path, audio, model.sr)
        print(f"Saved: {output_path}")

    print(f"\nGenerated {len(audios)} audio files!")


if __name__ == "__main__":
    main()
