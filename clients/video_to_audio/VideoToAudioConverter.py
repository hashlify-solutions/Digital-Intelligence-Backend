import os
import subprocess
from pathlib import Path
import logging

class VideoToAudioConverter:
    """
    A class to convert video files to audio files using FFmpeg.
    Supports all video formats defined in UfdrExtracter.
    """
    
    # Supported video formats from UfdrExtracter
    SUPPORTED_VIDEO_FORMATS = [
        ".mp4", ".mov", ".3gp", ".avi", ".webm", ".mkv", ".flv", ".wmv",
        ".asf", ".m4v", ".3g2", ".ts", ".mts", ".m2ts", ".vob", ".divx",
        ".xvid", ".rm", ".rmvb", ".ogv", ".f4v"
    ]
    
    def __init__(self, output_format="mp3", audio_bitrate="192k", audio_codec="mp3"):
        """
        Initialize the VideoToAudioConverter.
        
        Args:
            output_format (str): Output audio format (default: "mp3")
            audio_bitrate (str): Audio bitrate (default: "192k")
            audio_codec (str): Audio codec (default: "mp3")
        """
        self.output_format = output_format
        self.audio_bitrate = audio_bitrate
        self.audio_codec = audio_codec
        self.logger = logging.getLogger(__name__)
    
    def _validate_input_video(self, input_video_path):
        """
        Validate input video file.
        
        Args:
            input_video_path (str): Path to input video file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format is not supported
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video file not found: {input_video_path}")
        
        file_extension = Path(input_video_path).suffix.lower()
        if file_extension not in self.SUPPORTED_VIDEO_FORMATS:
            raise ValueError(f"Unsupported video format: {file_extension}. "
                           f"Supported formats: {', '.join(self.SUPPORTED_VIDEO_FORMATS)}")
    
    def _validate_output_path(self, output_audio_path):
        """
        Validate and prepare output audio path.
        
        Args:
            output_audio_path (str): Path for output audio file
            
        Returns:
            str: Validated output path with proper extension
            
        Raises:
            ValueError: If output directory is not writable
        """
        # Ensure output path has a proper audio extension
        if not output_audio_path.lower().endswith(('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac')):
            output_audio_path += f".{self.output_format}"
        
        output_dir = os.path.dirname(output_audio_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Check if directory is writable
        if output_dir and not os.access(output_dir, os.W_OK):
            raise ValueError(f"Output directory is not writable: {output_dir}")
            
        return output_audio_path
    
    def _check_ffmpeg_available(self):
        """
        Check if FFmpeg is available in the system.
        
        Returns:
            bool: True if FFmpeg is available, False otherwise
        """
        try:
            subprocess.run(["ffmpeg", "-version"], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def convert_video_to_audio(self, input_video_path, output_audio_path):
        """
        Convert video file to audio file.
        
        Args:
            input_video_path (str): Path to input video file
            output_audio_path (str): Path for output audio file
            
        Returns:
            str: Path to the output audio file if conversion is successful
            
        Raises:
            FileNotFoundError: If input video file doesn't exist
            ValueError: If video format is not supported or output path is invalid
            RuntimeError: If FFmpeg is not available or conversion fails
        """
        # Validate inputs
        self._validate_input_video(input_video_path)
        output_audio_path = self._validate_output_path(output_audio_path)
        
        # Check if FFmpeg is available
        if not self._check_ffmpeg_available():
            # raise RuntimeError("FFmpeg is not installed or not available in PATH. "
            #                  "Please install FFmpeg to use this converter.")
            print("FFmpeg is not installed or not available in PATH. Please install FFmpeg to use this converter.")
            return None
        
        try:
            # Determine output format from file extension
            output_ext = Path(output_audio_path).suffix.lower()
            
            # Construct FFmpeg command with format specification
            cmd = [
                "ffmpeg",
                "-i", input_video_path,          # Input video file
                "-vn",                           # Disable video recording
                "-acodec", self.audio_codec,     # Audio codec
                "-ab", self.audio_bitrate,       # Audio bitrate
                "-ar", "44100",                  # Audio sample rate
                "-y",                            # Overwrite output file
            ]
            
            # Add format specification if needed
            if output_ext == '.mp3':
                cmd.extend(["-f", "mp3"])
            elif output_ext == '.wav':
                cmd.extend(["-f", "wav"])
            elif output_ext == '.m4a':
                cmd.extend(["-f", "mp4"])
            elif output_ext == '.aac':
                cmd.extend(["-f", "adts"])
            elif output_ext == '.ogg':
                cmd.extend(["-f", "ogg"])
            elif output_ext == '.flac':
                cmd.extend(["-f", "flac"])
            
            cmd.append(output_audio_path)        # Output audio file
            
            self.logger.info(f"Converting video: {input_video_path} -> {output_audio_path}")
            self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            # Run FFmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                error_msg = f"FFmpeg conversion failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nError output: {result.stderr}"
                print(error_msg)
                return None
                # raise RuntimeError(error_msg)
            
            # Verify output file was created
            if not os.path.exists(output_audio_path):
                # raise RuntimeError("Conversion completed but output file was not created")
                print("Conversion completed but output file was not created")
                return None
            
            # Verify output file has content
            if os.path.getsize(output_audio_path) == 0:
                # raise RuntimeError("Conversion completed but output file is empty")
                print("Conversion completed but output file is empty")
                return None
            
            self.logger.info(f"Successfully converted video to audio: {output_audio_path}")
            return output_audio_path
            
        except subprocess.TimeoutExpired:
            # raise RuntimeError("Video conversion timed out after 5 minutes")
            print("Video conversion timed out after 5 minutes")
            return None
        except Exception as e:
            # Clean up partial output file if it exists
            if os.path.exists(output_audio_path):
                try:
                    os.remove(output_audio_path)
                except OSError:
                    pass
            print(f"Video conversion failed: {str(e)}")
            return None
            # raise RuntimeError(f"Video conversion failed: {str(e)}")
    
    def get_video_info(self, video_path):
        """
        Get information about a video file using FFprobe.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Video information including duration, format, etc.
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                return json.loads(result.stdout)
            else:
                return None
                
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            return None
