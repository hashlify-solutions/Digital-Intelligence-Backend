import whisper
import torch
import os
import logging
from typing import Optional, Dict, Any
import warnings
import subprocess
import shutil

# Suppress some warnings from whisper
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class Whisper:
    """
    Offline Whisper transcription client using OpenAI's Whisper model.
    Supports automatic language detection and works completely offline.
    """

    def __init__(self, model_size, device):
        """
        Initialize the Whisper transcription client.

        Args:
            model_size (str): Whisper model size - "tiny", "base", "small", "medium", "large"
            device (str): Device to use - "auto", "cpu", "cuda"
        """
        self.model_size = model_size
        self.device = device
        self.model = None

        logger.info(
            f"Initializing Whisper with model '{model_size}' on device '{self.device}'"
        )

        # Check for FFmpeg dependency
        self._check_ffmpeg_availability()
        
        # Load the model
        self._load_model()

    def _check_ffmpeg_availability(self):
        """
        Check if FFmpeg is available on the system.
        
        Raises:
            RuntimeError: If FFmpeg is not found
        """
        try:
            # Try to find ffmpeg in PATH
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                raise RuntimeError(
                    "FFmpeg not found. Please install FFmpeg: "
                    "sudo apt install ffmpeg (Ubuntu/Debian) or "
                    "brew install ffmpeg (macOS) or "
                    "choco install ffmpeg (Windows)"
                )
            
            # Test ffmpeg execution
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=120
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg found but not working properly: {result.stderr}")
                
            logger.info(f"FFmpeg found at: {ffmpeg_path}")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg found but timed out during version check")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg: "
                "sudo apt install ffmpeg (Ubuntu/Debian) or "
                "brew install ffmpeg (macOS) or "
                "choco install ffmpeg (Windows)"
            )

    def _load_model(self):
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper '{self.model_size}' model on {self.device}")

            # PyTorch 2.8+ compat: force CPU device context to avoid meta tensor initialization.
            # Load on CPU first to avoid unnecessary cross-device copies, then move to target device.
            with torch.device('cpu'):
                self.model = whisper.load_model(self.model_size, device='cpu')
            
            # Move to target device (e.g., 'cuda' for RTX 5090) in a single transfer
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            logger.info(
                f"Whisper '{self.model_size}' model loaded successfully on {self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load Whisper model on {self.device}: {str(e)}")
            # Fallback to CPU if CUDA fails
            if self.device != "cpu":
                logger.info("Retrying Whisper model loading on CPU")
                self.device = "cpu"
                with torch.device('cpu'):
                    self.model = whisper.load_model(self.model_size, device='cpu')
                logger.info(
                    f"Whisper '{self.model_size}' model loaded successfully on CPU"
                )
            else:
                raise

    def transcribe(
        self, audio_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.

        Args:
            audio_path (str): Path to the audio file
            language (str, optional): Language code (e.g., "en", "ar", "fr").
                                    If None, auto-detect language.

        Returns:
            Dict[str, Any]: Transcription result containing:
                - text: Full transcribed text
                - language: Detected/specified language
                - segments: List of text segments with timestamps
                - success: Boolean indicating success/failure
                - error: Error message if failed
        """
        try:
            # Validate audio file
            if not self._validate_audio_file(audio_path):
                return {
                    "text": "",
                    "language": None,
                    "segments": [],
                    "success": False,
                    "error": f"Invalid audio file: {audio_path}",
                }

            logger.info(f"Starting transcription of: {audio_path}")

            # Prepare transcription options
            options = {}
            if language:
                options["language"] = language
                logger.info(f"Using specified language: {language}")
            else:
                logger.info("Auto-detecting language")

            # Perform transcription
            result = self.model.transcribe(audio_path, **options)

            # Extract key information
            transcribed_text = result.get("text", "").strip()
            detected_language = result.get("language", "unknown")
            segments = result.get("segments", [])

            logger.info(
                f"Transcription completed. Language: {detected_language}, "
                f"Text length: {len(transcribed_text)} characters"
            )

            return {
                "text": transcribed_text,
                "language": detected_language,
                "segments": segments,
                "success": True,
                "error": None,
            }

        except FileNotFoundError as e:
            if "ffmpeg" in str(e).lower():
                error_msg = f"FFmpeg not found. Please install FFmpeg to enable audio transcription: {str(e)}"
            else:
                error_msg = f"File not found error transcribing audio {audio_path}: {str(e)}"
            logger.error(error_msg)
            return {
                "text": "",
                "language": None,
                "segments": [],
                "success": False,
                "error": error_msg,
            }
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error transcribing audio {audio_path}: {str(e)}"
            logger.error(error_msg)
            return {
                "text": "",
                "language": None,
                "segments": [],
                "success": False,
                "error": error_msg,
            }
        except Exception as e:
            error_msg = f"Error transcribing audio {audio_path}: {str(e)}"
            logger.error(error_msg)
            return {
                "text": "",
                "language": None,
                "segments": [],
                "success": False,
                "error": error_msg,
            }

    def transcribe_with_timestamps(
        self, audio_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file with detailed timestamp information.

        Args:
            audio_path (str): Path to the audio file
            language (str, optional): Language code. If None, auto-detect.

        Returns:
            Dict[str, Any]: Transcription result with detailed segments
        """
        result = self.transcribe(audio_path, language)

        if not result["success"]:
            return result

        # Format segments with detailed timestamps
        formatted_segments = []
        for segment in result["segments"]:
            formatted_segments.append(
                {
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "").strip(),
                    "duration": segment.get("end", 0) - segment.get("start", 0),
                }
            )

        result["formatted_segments"] = formatted_segments
        return result

    def _validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate that the audio file exists and has a supported format.

        Args:
            audio_path (str): Path to the audio file

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return False

            # Check file size (should not be empty)
            if os.path.getsize(audio_path) == 0:
                logger.error(f"Audio file is empty: {audio_path}")
                return False

            # Check file extension (Whisper supports many formats)
            supported_extensions = {
                ".amr",
                ".mp3",
                ".wav",
                ".m4a",
                ".aac",
                ".wma",
                ".ogg",
                ".3ga",
                ".awb",
                ".flac",
                ".opus",
                ".gsm",
                ".qcp",
                ".evrc",
                ".amr-wb",
                ".amr-nb",
                ".evs",
                ".silk",
                ".speex",
                ".vorbis",
                ".ac3",
                ".eac3",
                ".dts",
                ".pcm",
                ".aiff",
                ".au",
                ".snd",
                ".caf",
                ".adts",
                ".mp2",
                ".mpa",
                ".ra",
                ".wv",
                ".tta",
                ".ape",
                ".mka",
            }

            file_extension = os.path.splitext(audio_path)[1].lower()
            if file_extension not in supported_extensions:
                logger.warning(
                    f"Unusual audio format: {file_extension}. Whisper will attempt to process it."
                )

            return True

        except Exception as e:
            logger.error(f"Error validating audio file {audio_path}: {str(e)}")
            return False

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of languages supported by Whisper.

        Returns:
            Dict[str, str]: Language code to language name mapping
        """
        return {
            "en": "English",
            "zh": "Chinese",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "ko": "Korean",
            "fr": "French",
            "ja": "Japanese",
            "pt": "Portuguese",
            "tr": "Turkish",
            "pl": "Polish",
            "ca": "Catalan",
            "nl": "Dutch",
            "ar": "Arabic",
            "sv": "Swedish",
            "it": "Italian",
            "id": "Indonesian",
            "hi": "Hindi",
            "fi": "Finnish",
            "vi": "Vietnamese",
            "he": "Hebrew",
            "uk": "Ukrainian",
            "el": "Greek",
            "ms": "Malay",
            "cs": "Czech",
            "ro": "Romanian",
            "da": "Danish",
            "hu": "Hungarian",
            "ta": "Tamil",
            "no": "Norwegian",
            "th": "Thai",
            "ur": "Urdu",
            "hr": "Croatian",
            "bg": "Bulgarian",
            "lt": "Lithuanian",
            "la": "Latin",
            "mi": "Maori",
            "ml": "Malayalam",
            "cy": "Welsh",
            "sk": "Slovak",
            "te": "Telugu",
            "fa": "Persian",
            "lv": "Latvian",
            "bn": "Bengali",
            "sr": "Serbian",
            "az": "Azerbaijani",
            "sl": "Slovenian",
            "kn": "Kannada",
            "et": "Estonian",
            "mk": "Macedonian",
            "br": "Breton",
            "eu": "Basque",
            "is": "Icelandic",
            "hy": "Armenian",
            "ne": "Nepali",
            "mn": "Mongolian",
            "bs": "Bosnian",
            "kk": "Kazakh",
            "sq": "Albanian",
            "sw": "Swahili",
            "gl": "Galician",
            "mr": "Marathi",
            "pa": "Punjabi",
            "si": "Sinhala",
            "km": "Khmer",
            "sn": "Shona",
            "yo": "Yoruba",
            "so": "Somali",
            "af": "Afrikaans",
            "oc": "Occitan",
            "ka": "Georgian",
            "be": "Belarusian",
            "tg": "Tajik",
            "sd": "Sindhi",
            "gu": "Gujarati",
            "am": "Amharic",
            "yi": "Yiddish",
            "lo": "Lao",
            "uz": "Uzbek",
            "fo": "Faroese",
            "ht": "Haitian Creole",
            "ps": "Pashto",
            "tk": "Turkmen",
            "nn": "Nynorsk",
            "mt": "Maltese",
            "sa": "Sanskrit",
            "lb": "Luxembourgish",
            "my": "Myanmar",
            "bo": "Tibetan",
            "tl": "Tagalog",
            "mg": "Malagasy",
            "as": "Assamese",
            "tt": "Tatar",
            "haw": "Hawaiian",
            "ln": "Lingala",
            "ha": "Hausa",
            "ba": "Bashkir",
            "jw": "Javanese",
            "su": "Sundanese",
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded Whisper model.

        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "supports_offline": True,
            "supports_timestamps": True,
            "supports_language_detection": True,
            "supported_languages": len(self.get_supported_languages()),
            "model_loaded": self.model is not None,
        }
