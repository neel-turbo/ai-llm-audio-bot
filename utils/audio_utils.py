import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import requests
from gtts import gTTS
import os
from typing import Optional

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio processor
        
        Args:
            sample_rate: Sampling rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
        self.temp_dir = tempfile.mkdtemp()

    def record_audio(self, duration: int = 5) -> Optional[str]:
        """
        Record audio
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Path to the recorded file
        """
        try:
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1
            )
            sd.wait()  # Wait for recording to complete
            
            # Save the recorded audio
            temp_path = os.path.join(self.temp_dir, 'recording.wav')
            sf.write(temp_path, recording, self.sample_rate)
            
            return temp_path
        except Exception as e:
            print(f"Recording error: {str(e)}")
            return None

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe using Groq's hosted API; no local speech model is loaded.

        Reads GROQ_API from the environment (GROQ_API_KEY is also accepted).
        Raises RuntimeError with a user-facing explanation on failure.
        """
        api_key = os.getenv("GROQ_API") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API in your .env file and restart Streamlit to enable transcription.")

        try:
            with open(audio_path, "rb") as audio_file:
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (os.path.basename(audio_path), audio_file)},
                    data={"model": "whisper-large-v3-turbo", "response_format": "json"},
                    timeout=(10, 120),
                )
            if response.status_code == 401:
                raise RuntimeError("Groq rejected your API key. Check GROQ_API in .env and restart Streamlit.")
            if response.status_code == 429:
                raise RuntimeError("Groq's rate limit or quota was reached. Check your Groq account limits and try again later.")
            if response.status_code == 413:
                raise RuntimeError("The audio file is too large for Groq. Try a shorter recording or a smaller file.")
            if response.status_code == 400:
                raise RuntimeError("Groq could not process this audio. Try a valid WAV or MP3 recording.")
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError("Groq transcription timed out. Please try again.") from exc
        except requests.ConnectionError as exc:
            raise RuntimeError("Cannot connect to Groq. Check your internet connection.") from exc
        except requests.RequestException as exc:
            raise RuntimeError("Groq transcription failed. Please try again or check Groq's service status.") from exc
        except OSError as exc:
            raise RuntimeError("Cannot read the audio file. Please record or upload it again.") from exc

        try:
            text = response.json()["text"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Groq returned an invalid transcription response. Please try again.") from exc
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("No speech was detected. Please record again and speak clearly.")
        return text.strip()

    def text_to_speech(self, text: str, lang: str = 'en') -> Optional[str]:
        """
        Convert text to speech
        
        Args:
            text: Text to convert
            lang: Language code (default: English)
            
        Returns:
            Path to generated audio file
        """
        try:
            tts = gTTS(text=text, lang=lang)
            temp_path = os.path.join(self.temp_dir, 'response.mp3')
            tts.save(temp_path)
            return temp_path
        except Exception as e:
            print(f"Text-to-speech error: {str(e)}")
            return None

    def preprocess_audio(self, audio_path: str) -> Optional[str]:
        """
        Preprocess audio file (resampling, noise reduction, etc.)
        
        Args:
            audio_path: Input audio file path
            
        Returns:
            Path to processed audio file
        """
        try:
            # Read audio file
            data, sample_rate = sf.read(audio_path)
            
            # Convert stereo to mono if needed
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Resample to target sample rate
            if sample_rate != self.sample_rate:
                # Resampling logic can be added here
                pass
            
            # Save processed audio
            processed_path = os.path.join(self.temp_dir, 'processed_audio.wav')
            sf.write(processed_path, data, self.sample_rate)
            
            return processed_path
        except Exception as e:
            print(f"Audio preprocessing error: {str(e)}")
            return None

    def cleanup(self):
        """
        Clean up temporary files
        """
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning temporary files: {str(e)}")