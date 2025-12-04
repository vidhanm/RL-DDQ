"""
Voice Integration Module with LiveKit
Real-time voice conversations with the debt collection agent

Components:
1. VoiceAgent - Main voice conversation handler
2. STTHandler - Speech-to-text processing  
3. TTSHandler - Text-to-speech synthesis
4. AudioPipeline - Streaming audio management

Requires:
- livekit
- livekit-agents
- deepgram-sdk (for STT, optional)
- elevenlabs (for TTS, optional)
"""

import asyncio
import json
import os
import time
from typing import Optional, Callable, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

class VoiceState(Enum):
    """Voice agent state"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class VoiceConfig:
    """Configuration for voice integration"""
    # LiveKit
    livekit_url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")
    
    # STT (Speech-to-Text)
    stt_provider: str = "whisper"  # whisper, deepgram, azure
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    whisper_model: str = "base"  # tiny, base, small, medium, large
    
    # TTS (Text-to-Speech)
    tts_provider: str = "local"  # local, elevenlabs, azure
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100
    
    # Timeouts
    silence_timeout_sec: float = 2.0
    max_listen_sec: float = 30.0
    response_timeout_sec: float = 10.0


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech"""
    text: str
    timestamp: float
    is_final: bool = False
    confidence: float = 1.0


@dataclass
class VoiceResponse:
    """Response from voice agent"""
    text: str
    audio_data: Optional[bytes] = None
    action_taken: Optional[str] = None
    q_values: Optional[Dict[str, float]] = None


# ============================================================================
# ABSTRACT HANDLERS
# ============================================================================

class BaseSTTHandler(ABC):
    """Abstract speech-to-text handler"""
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> TranscriptSegment:
        """Transcribe audio to text"""
        pass
    
    @abstractmethod
    async def start_stream(self):
        """Start streaming transcription"""
        pass
    
    @abstractmethod
    async def process_audio_chunk(self, chunk: bytes) -> Optional[TranscriptSegment]:
        """Process streaming audio chunk"""
        pass
    
    @abstractmethod
    async def stop_stream(self) -> TranscriptSegment:
        """Stop streaming and get final transcript"""
        pass


class BaseTTSHandler(ABC):
    """Abstract text-to-speech handler"""
    
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio"""
        pass
    
    @abstractmethod
    async def stream_synthesize(self, text: str):
        """Stream synthesized audio chunks"""
        pass


# ============================================================================
# WHISPER STT (LOCAL)
# ============================================================================

class WhisperSTTHandler(BaseSTTHandler):
    """Local Whisper-based speech-to-text"""
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self._audio_buffer = bytearray()
        self._stream_active = False
        
    def _load_model(self):
        """Lazy load Whisper model"""
        if self.model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.model = whisper.load_model(self.model_name)
            except ImportError:
                logger.error("Whisper not installed. Run: pip install openai-whisper")
                raise
    
    async def transcribe(self, audio_data: bytes) -> TranscriptSegment:
        """Transcribe audio bytes to text"""
        self._load_model()
        
        import numpy as np
        import tempfile
        import wave
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Save to temp file (Whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            with wave.open(f, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio_data)
        
        try:
            result = self.model.transcribe(temp_path)
            return TranscriptSegment(
                text=result["text"].strip(),
                timestamp=time.time(),
                is_final=True,
                confidence=1.0  # Whisper doesn't provide confidence
            )
        finally:
            os.unlink(temp_path)
    
    async def start_stream(self):
        """Start buffering audio"""
        self._audio_buffer = bytearray()
        self._stream_active = True
    
    async def process_audio_chunk(self, chunk: bytes) -> Optional[TranscriptSegment]:
        """Buffer audio chunk"""
        if self._stream_active:
            self._audio_buffer.extend(chunk)
        return None  # Whisper doesn't support streaming
    
    async def stop_stream(self) -> TranscriptSegment:
        """Process buffered audio"""
        self._stream_active = False
        if len(self._audio_buffer) > 0:
            return await self.transcribe(bytes(self._audio_buffer))
        return TranscriptSegment(text="", timestamp=time.time(), is_final=True)


# ============================================================================
# DEEPGRAM STT (CLOUD)
# ============================================================================

class DeepgramSTTHandler(BaseSTTHandler):
    """Deepgram cloud-based speech-to-text with streaming support"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._connection = None
        self._final_transcript = ""
        self._interim_transcript = ""
        
    async def transcribe(self, audio_data: bytes) -> TranscriptSegment:
        """Transcribe audio using Deepgram REST API"""
        try:
            from deepgram import Deepgram
        except ImportError:
            logger.error("Deepgram not installed. Run: pip install deepgram-sdk")
            raise
        
        dg = Deepgram(self.api_key)
        
        source = {'buffer': audio_data, 'mimetype': 'audio/wav'}
        response = await dg.transcription.prerecorded(
            source,
            {'punctuate': True, 'language': 'en-US'}
        )
        
        text = response['results']['channels'][0]['alternatives'][0]['transcript']
        confidence = response['results']['channels'][0]['alternatives'][0]['confidence']
        
        return TranscriptSegment(
            text=text,
            timestamp=time.time(),
            is_final=True,
            confidence=confidence
        )
    
    async def start_stream(self):
        """Start Deepgram streaming connection"""
        try:
            from deepgram import Deepgram
        except ImportError:
            raise ImportError("deepgram-sdk required for streaming")
        
        dg = Deepgram(self.api_key)
        
        async def on_message(self_ws, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if result.is_final:
                self._final_transcript += transcript + " "
            else:
                self._interim_transcript = transcript
        
        options = {
            'punctuate': True,
            'interim_results': True,
            'language': 'en-US'
        }
        
        self._connection = await dg.transcription.live(options)
        self._connection.registerHandler('Results', on_message)
        self._final_transcript = ""
        self._interim_transcript = ""
    
    async def process_audio_chunk(self, chunk: bytes) -> Optional[TranscriptSegment]:
        """Send audio chunk to Deepgram"""
        if self._connection:
            await self._connection.send(chunk)
            
            if self._interim_transcript:
                return TranscriptSegment(
                    text=self._interim_transcript,
                    timestamp=time.time(),
                    is_final=False
                )
        return None
    
    async def stop_stream(self) -> TranscriptSegment:
        """Close connection and get final transcript"""
        if self._connection:
            await self._connection.finish()
            self._connection = None
        
        return TranscriptSegment(
            text=self._final_transcript.strip(),
            timestamp=time.time(),
            is_final=True
        )


# ============================================================================
# LOCAL TTS (pyttsx3)
# ============================================================================

class LocalTTSHandler(BaseTTSHandler):
    """Local text-to-speech using pyttsx3"""
    
    def __init__(self, rate: int = 150, voice_idx: int = 0):
        self.rate = rate
        self.voice_idx = voice_idx
        self._engine = None
    
    def _get_engine(self):
        """Lazy load pyttsx3 engine"""
        if self._engine is None:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', self.rate)
                voices = self._engine.getProperty('voices')
                if self.voice_idx < len(voices):
                    self._engine.setProperty('voice', voices[self.voice_idx].id)
            except ImportError:
                logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
                raise
        return self._engine
    
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes"""
        import tempfile
        import wave
        
        engine = self._get_engine()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            with open(temp_path, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def stream_synthesize(self, text: str):
        """Stream audio in chunks (not truly streaming for pyttsx3)"""
        audio_data = await self.synthesize(text)
        chunk_size = 4096
        
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size]


# ============================================================================
# ELEVENLABS TTS (CLOUD)
# ============================================================================

class ElevenLabsTTSHandler(BaseTTSHandler):
    """ElevenLabs cloud text-to-speech with streaming"""
    
    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.base_url = "https://api.elevenlabs.io/v1"
    
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text using ElevenLabs API"""
        import aiohttp
        
        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    error = await response.text()
                    raise Exception(f"ElevenLabs API error: {error}")
    
    async def stream_synthesize(self, text: str):
        """Stream audio from ElevenLabs"""
        import aiohttp
        
        url = f"{self.base_url}/text-to-speech/{self.voice_id}/stream"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                async for chunk in response.content.iter_chunked(1024):
                    yield chunk


# ============================================================================
# VOICE AGENT (MAIN ORCHESTRATOR)
# ============================================================================

class VoiceAgent:
    """
    Main voice agent that orchestrates STT, RL agent, and TTS
    
    Flow:
    1. Listen for user speech → STT → text
    2. Pass text to RL agent → get response + action
    3. Convert response to speech → TTS → play audio
    """
    
    def __init__(
        self,
        rl_agent,  # DDQ or DQN agent
        state_encoder,
        env,
        config: VoiceConfig = None,
        on_state_change: Optional[Callable[[VoiceState], None]] = None
    ):
        self.rl_agent = rl_agent
        self.state_encoder = state_encoder
        self.env = env
        self.config = config or VoiceConfig()
        self.on_state_change = on_state_change
        
        # State
        self.state = VoiceState.IDLE
        self.conversation_history: List[Dict[str, str]] = []
        self.current_episode_step = 0
        
        # Initialize handlers
        self.stt_handler = self._create_stt_handler()
        self.tts_handler = self._create_tts_handler()
        
        logger.info("VoiceAgent initialized")
    
    def _create_stt_handler(self) -> BaseSTTHandler:
        """Create STT handler based on config"""
        if self.config.stt_provider == "deepgram":
            return DeepgramSTTHandler(self.config.deepgram_api_key)
        else:  # whisper
            return WhisperSTTHandler(self.config.whisper_model)
    
    def _create_tts_handler(self) -> BaseTTSHandler:
        """Create TTS handler based on config"""
        if self.config.tts_provider == "elevenlabs":
            return ElevenLabsTTSHandler(
                self.config.elevenlabs_api_key,
                self.config.elevenlabs_voice_id
            )
        else:  # local
            return LocalTTSHandler()
    
    def _set_state(self, new_state: VoiceState):
        """Update state and notify callback"""
        self.state = new_state
        if self.on_state_change:
            self.on_state_change(new_state)
        logger.info(f"Voice state: {new_state.value}")
    
    async def process_speech(self, audio_data: bytes) -> VoiceResponse:
        """
        Process speech input and generate response
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            VoiceResponse with text and optional audio
        """
        try:
            # Step 1: Speech to Text
            self._set_state(VoiceState.LISTENING)
            transcript = await self.stt_handler.transcribe(audio_data)
            
            if not transcript.text.strip():
                return VoiceResponse(text="I didn't catch that. Could you repeat?")
            
            logger.info(f"User said: {transcript.text}")
            
            # Step 2: Process with RL Agent
            self._set_state(VoiceState.PROCESSING)
            response = await self._process_with_agent(transcript.text)
            
            # Step 3: Text to Speech
            self._set_state(VoiceState.SPEAKING)
            audio_response = await self.tts_handler.synthesize(response.text)
            response.audio_data = audio_response
            
            self._set_state(VoiceState.IDLE)
            return response
            
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            self._set_state(VoiceState.ERROR)
            return VoiceResponse(text="Sorry, there was an error processing your message.")
    
    async def _process_with_agent(self, user_text: str) -> VoiceResponse:
        """Process user input with RL agent"""
        import torch
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_text})
        
        # Encode state
        state_dict = {
            "conversation_history": self.conversation_history,
            "step": self.current_episode_step,
            "persona": self.env.persona.type if hasattr(self.env, 'persona') else "unknown"
        }
        
        state_tensor = torch.FloatTensor(
            self.state_encoder.encode(state_dict)
        ).unsqueeze(0)
        
        # Get action from agent
        action, q_values = self.rl_agent.select_action(state_tensor, return_q=True)
        
        # Map action to response
        action_names = [
            "greeting", "ask_situation", "empathy", 
            "payment_options", "consequence_warning", "close"
        ]
        action_name = action_names[action] if action < len(action_names) else "unknown"
        
        # Generate response based on action
        response_templates = {
            0: "Hello! Thank you for taking my call. I'm here to discuss your account with you.",
            1: "Could you tell me a bit about your current situation?",
            2: "I understand this can be difficult. We're here to help you through this.",
            3: "We have several payment options available. Would you like to hear about them?",
            4: "I want to make sure you're aware of the timeline for this account.",
            5: "Thank you for speaking with me today. Let's summarize what we discussed."
        }
        
        response_text = response_templates.get(
            action, 
            "How can I help you with your account today?"
        )
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        self.current_episode_step += 1
        
        # Format Q-values
        q_value_dict = {
            action_names[i]: q_values[0][i].item() 
            for i in range(min(len(action_names), q_values.shape[1]))
        } if q_values is not None else None
        
        return VoiceResponse(
            text=response_text,
            action_taken=action_name,
            q_values=q_value_dict
        )
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_history = []
        self.current_episode_step = 0
        self._set_state(VoiceState.IDLE)
        logger.info("Conversation reset")


# ============================================================================
# LIVEKIT INTEGRATION
# ============================================================================

class LiveKitVoiceRoom:
    """
    LiveKit room handler for real-time voice streaming
    
    Manages WebRTC connection to LiveKit server for:
    - Receiving audio from user
    - Sending agent audio responses
    - Real-time transcription and TTS
    """
    
    def __init__(
        self,
        voice_agent: VoiceAgent,
        config: VoiceConfig = None
    ):
        self.voice_agent = voice_agent
        self.config = config or VoiceConfig()
        self._room = None
        self._audio_buffer = bytearray()
        self._is_listening = False
        
    async def connect(self, room_name: str, participant_identity: str = "agent"):
        """Connect to a LiveKit room"""
        try:
            from livekit import rtc
        except ImportError:
            logger.error("LiveKit not installed. Run: pip install livekit")
            raise
        
        # Create room
        self._room = rtc.Room()
        
        # Register event handlers
        @self._room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.create_task(self._handle_audio_track(track))
        
        @self._room.on("participant_disconnected")
        def on_participant_left(participant):
            logger.info(f"Participant left: {participant.identity}")
        
        # Connect to room
        token = self._generate_token(room_name, participant_identity)
        await self._room.connect(self.config.livekit_url, token)
        
        logger.info(f"Connected to LiveKit room: {room_name}")
    
    def _generate_token(self, room_name: str, identity: str) -> str:
        """Generate access token for LiveKit"""
        try:
            from livekit import api
        except ImportError:
            raise ImportError("livekit-api required for token generation")
        
        token = api.AccessToken(
            self.config.livekit_api_key,
            self.config.livekit_api_secret
        )
        token.with_identity(identity)
        token.with_name("Debt Collection Agent")
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name
        ))
        
        return token.to_jwt()
    
    async def _handle_audio_track(self, track):
        """Process incoming audio track"""
        from livekit import rtc
        
        audio_stream = rtc.AudioStream(track)
        silence_start = None
        
        async for event in audio_stream:
            # Buffer audio
            self._audio_buffer.extend(event.frame.data)
            
            # Detect silence for end-of-speech
            is_silent = self._is_silence(event.frame.data)
            
            if is_silent:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > self.config.silence_timeout_sec:
                    # User stopped speaking
                    if len(self._audio_buffer) > 1000:  # Minimum audio length
                        await self._process_audio_buffer()
                    silence_start = None
            else:
                silence_start = None
    
    def _is_silence(self, audio_data: bytes, threshold: int = 500) -> bool:
        """Check if audio frame is silence"""
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return np.abs(audio_array).mean() < threshold
    
    async def _process_audio_buffer(self):
        """Process buffered audio and respond"""
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer = bytearray()
        
        # Process with voice agent
        response = await self.voice_agent.process_speech(audio_data)
        
        # Publish audio response
        if response.audio_data and self._room:
            await self._publish_audio(response.audio_data)
    
    async def _publish_audio(self, audio_data: bytes):
        """Publish audio to LiveKit room"""
        from livekit import rtc
        
        # Create audio source
        audio_source = rtc.AudioSource(16000, 1)
        
        # Publish track
        track = rtc.LocalAudioTrack.create_audio_track("agent_audio", audio_source)
        await self._room.local_participant.publish_track(track)
        
        # Send audio frames
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        frame_size = 480  # 30ms at 16kHz
        
        for i in range(0, len(audio_array), frame_size):
            frame_data = audio_array[i:i + frame_size]
            if len(frame_data) < frame_size:
                frame_data = np.pad(frame_data, (0, frame_size - len(frame_data)))
            
            frame = rtc.AudioFrame(
                data=frame_data.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=frame_size
            )
            await audio_source.capture_frame(frame)
    
    async def disconnect(self):
        """Disconnect from LiveKit room"""
        if self._room:
            await self._room.disconnect()
            self._room = None
            logger.info("Disconnected from LiveKit room")


# ============================================================================
# CLI FOR TESTING
# ============================================================================

def create_voice_demo():
    """Create a simple voice demo script"""
    demo_script = '''
"""
Voice Demo - Test voice integration locally
Run: python voice_demo.py

Requirements:
  pip install openai-whisper pyttsx3 sounddevice scipy
"""

import asyncio
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as write_wav
import tempfile
import os

# Import from your project
from voice_integration import VoiceAgent, VoiceConfig, LocalTTSHandler, WhisperSTTHandler


def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Record audio from microphone"""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    print("Recording complete.")
    return audio.tobytes()


async def main():
    # Create dummy agent for testing
    class DummyAgent:
        def select_action(self, state, return_q=False):
            import torch
            action = 0  # greeting
            q_values = torch.randn(1, 6) if return_q else None
            return action, q_values
    
    class DummyEncoder:
        def encode(self, state_dict):
            return [0.0] * 20
    
    class DummyEnv:
        class Persona:
            type = "cooperative"
        persona = Persona()
    
    # Initialize voice agent
    config = VoiceConfig(stt_provider="whisper", tts_provider="local")
    voice_agent = VoiceAgent(
        rl_agent=DummyAgent(),
        state_encoder=DummyEncoder(),
        env=DummyEnv(),
        config=config
    )
    
    print("Voice Demo Started!")
    print("=" * 40)
    
    while True:
        input("Press Enter to start recording (or Ctrl+C to exit)...")
        
        # Record audio
        audio_data = record_audio(duration=5.0)
        
        # Process with voice agent
        response = await voice_agent.process_speech(audio_data)
        
        print(f"\\nAgent Response: {response.text}")
        if response.action_taken:
            print(f"Action Taken: {response.action_taken}")
        
        print("-" * 40)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nExiting...")
'''
    return demo_script


if __name__ == "__main__":
    # Print demo script when run directly
    print(create_voice_demo())
