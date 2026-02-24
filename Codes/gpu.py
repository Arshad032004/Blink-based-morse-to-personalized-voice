import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
import torchaudio

# Suppress warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure for best performance
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# Initialize TTS with proper device handling
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tts = TextToSpeech(device=device)

# Verify resources
if device == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available memory: {torch.cuda.mem_get_info()[0] / 1024 ** 2:.1f} MB")
else:
    print("Warning: Using CPU - Generation will be very slow")

# Load voice samples (must be 16kHz mono WAV)
try:
    voice_samples = [load_audio(f"{i}.wav", 22050) for i in [5, 6, 7]]
except Exception as e:
    print(f"Error loading samples: {e}")
    print("Please ensure:")
    print("1. Files exist (sample_1.wav, sample_2.wav, sample_3.wav)")
    print("2. They are 16kHz mono WAV format")
    exit()

# Text to synthesize
text = "This is my cloned voice running on my system."

# Generation with error handling
try:
    with torch.inference_mode():
        gen = tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            preset="ultra_fast",  # Must use this for limited resources
            k=1,  # Single output
            num_autoregressive_samples=1,
            diffusion_iterations=20,  # Reduced from default 30
            length_penalty=1.0,
            temperature=0.8,
            # Removed 'half' parameter as it causes errors
        )

    torchaudio.save("output.wav", gen.squeeze(0).cpu(), 24000)
    print("Success! Saved to output.wav")

except Exception as e:
    print(f"Generation failed: {e}")
    if device == 'cuda' and "CUDA out of memory" in str(e):
        print("\nGPU OUT OF MEMORY SOLUTIONS:")
        print("1. Close all other applications")
        print("2. Restart your Python kernel")
        print("3. Try this CPU fallback version:")
        print("   tts = TextToSpeech(device='cpu')")
finally:
    if device == 'cuda':
        torch.cuda.empty_cache()