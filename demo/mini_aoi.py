"""
Async Artificial Open Intelligence (Async-AOI)

A minimal AI system with async architecture featuring:
- Audio recording and transcription
- LLM-powered responses with conversation memory
- Text-to-speech output
- Event-driven communication between components
- Console UI with real-time status updates

Usage: PYTHONPATH=. python demo/async_aoi.py
"""

# Debug logging switch - set to False to disable all logging messages (production mode)
DEBUG_LOGGING = True
INPUT_MODE = "keyboard" # Options: keyboard, voice
OUTPUT_MODE = "console" # Options: console, speaker

OPENAI_API_KEY_FILE = "configs/openai_api_key"
SYSTEM_MSG_EN = "You are AOI (pronounced as ah-o-e), a LLM-driven personal assistant built by Jinghong Chen. Your response should be oral and brief."
SYSTEM_MSG_ZH = "你叫小蓝，是一个由大语言模型驱动的个人助理. 你的作者是陈镜鸿。你的回答应该口语化，简洁明了。"

# Language switch - set to "ZH" for Chinese, "EN" for English
LANGUAGE = "EN" # Options: ZH, EN

import asyncio
import logging
import json
import datetime
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Set
from abc import abstractmethod
import openai
import pyaudio
import wave
from pynput import keyboard
import tempfile
from enum import Enum, auto
import queue
import time
import threading
import platform
import sys
import subprocess
import requests

# Import the dialogue engine
sys.path.append('src')
from aoi.dialogue_engine import AOIDialogueEngine

""" ===================== Hyper-Parameters ===================== """
DEBUG_LOGGING = True
LANGUAGE = "ZH"

""" ===================== Logging ===================== """
# Set up logging with configurable debug level
log_level = logging.DEBUG if DEBUG_LOGGING else logging.ERROR
logging.basicConfig(
    filename='aoi_async.log',
    level=log_level,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
    
# Create console handler for real-time output
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(console_formatter)

# Get the root logger and add console handler
logger = logging.getLogger()
logger.addHandler(console_handler)

logger.info("Mini-AOI starting up...")
logger.debug("Debug logging enabled")

""" ===================== Events ===================== """

class EventType(Enum):
    """Event types in the system"""
    RECORDING_START = auto()
    RECORDING_STOP = auto()
    TRANSCRIPTION = auto()
    LLM_RESPONSE = auto()
    ERROR = auto()
    STATUS_UPDATE = auto()
    TTS_START = auto()
    TTS_COMPLETE = auto()

@dataclass
class Event:
    """Event class for system-wide communication"""
    type: EventType
    source: str
    content: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().timestamp()

    def to_dict(self) -> dict:
        return {
            'type': self.type.name,
            'source': self.source,
            'content': self.content,
            'timestamp': self.timestamp
        }

class EventBus:
    """Async event bus for component communication"""
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._subscribers: Dict[EventType, Set[str]] = {event_type: set() for event_type in EventType}
        self._logger = logging.getLogger('EventBus')

    def register_queue(self, component: str) -> asyncio.Queue:
        """Register a new component's queue"""
        if component not in self._queues:
            self._queues[component] = asyncio.Queue()
            self._logger.info(f"Registered queue for {component}")
        return self._queues[component]

    def subscribe(self, component: str, event_types: Set[EventType]):
        """Subscribe a component to specific event types"""
        for event_type in event_types:
            self._subscribers[event_type].add(component)
        self._logger.info(f"{component} subscribed to {[et.name for et in event_types]}")

    async def publish(self, event: Event):
        """Publish an event to all subscribed components"""
        self._logger.debug(f"Publishing {event.type.name} from {event.source}")
        for component in self._subscribers[event.type]:
            if component != event.source:  # Don't send back to source
                await self._queues[component].put(event)
                self._logger.debug(f"Sent {event.type.name} to {component}")

    async def emit(self, event: Event):
        """Emit an event to all subscribed components"""
        await self.publish(event)

""" ===================== General AOI Module ===================== """
class AOIModule:
    def __init__(self, name, event_bus, event_types_to_subscribe):
        self.name = name
        self.event_bus = event_bus
        self._logger = logging.getLogger(self.name)
        self._queue = self.event_bus.register_queue(self.name)
        self.event_bus.subscribe(self.name, event_types_to_subscribe)

    @abstractmethod
    async def run(self):
        """Main event processing loop"""
        pass


""" ===================== I/O Modules ===================== """

class AudioRecorder:
    """Handles audio recording using PyAudio"""
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.recording = False
        self.frames = []
        self.p = None
        self.stream = None
        self._logger = logging.getLogger('AudioRecorder')
        self._record_thread = None

    def start_recording(self):
        """Start recording audio"""
        if self.recording:
            return
            
        self.recording = True
        self.frames = []
        self._logger.info("Starting audio recording")
        
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            def record():
                while self.recording and self.stream:
                    try:
                        data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                        self.frames.append(data)
                    except Exception as e:
                        self._logger.error(f"Error reading audio: {e}")
                        break
            
            self._record_thread = threading.Thread(target=record)
            self._record_thread.daemon = True
            self._record_thread.start()
            self._logger.info("Audio recording thread started")
            
        except Exception as e:
            self._logger.error(f"Error starting recording: {e}")
            self.cleanup()
            raise

    def stop_recording(self):
        """Stop recording and save to file"""
        if not self.recording:
            self._logger.warning("Attempted to stop recording when not recording")
            return None
            
        self.recording = False
        self._logger.info("Stopping audio recording")
        
        if self._record_thread:
            self._record_thread.join(timeout=1.0)
        
        if not self.frames:
            self._logger.warning("No audio frames recorded")
            return None
            
        try:
            # Create a timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            
            # Save the recording
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            self._logger.info(f"Audio saved to {filename}")
            print(f"\nAudio saved to {filename} for inspection")
            return filename
            
        except Exception as e:
            self._logger.error(f"Error saving recording: {e}")
            return None
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self._logger.error(f"Error closing stream: {e}")
            self.stream = None
            
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                self._logger.error(f"Error terminating PyAudio: {e}")
            self.p = None
            
        self.recording = False
        self.frames = []

class AOIAudioListener(AOIModule):
    """Handles audio recording and transcription"""
    def __init__(self, event_bus: EventBus, model_name: str = "gpt-4o-mini-transcribe"):
        event_types_to_subscribe = {
            EventType.STATUS_UPDATE
        }
        super().__init__("AudioListener", event_bus, event_types_to_subscribe)
        self.recorder = AudioRecorder()
        self.model_name = model_name
        self._keyboard_listener = None
        self._pressed = False
        self._key_queue = queue.Queue()  # Use synchronous queue for keyboard events
        self._init_model()
        self._loop = None  # Store the event loop

    def _init_model(self):
        """Initialize OpenAI client"""
        try:
            with open(OPENAI_API_KEY_FILE, 'r') as f:
                api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key file is empty")
            self.client = openai.OpenAI(api_key=api_key)
            self._logger.info("OpenAI client initialized")
        except Exception as e:
            self._logger.error(f"Error initializing OpenAI client: {e}")
            raise

    def _on_key_press(self, key):
        """Handle key press events - synchronous callback"""
        try:
            if key == keyboard.Key.space:
                self._logger.debug(f"Space key pressed, current state: pressed={self._pressed}")
                if not self._pressed:
                    self._pressed = True
                    self._key_queue.put(('press', None))
                    self._logger.debug("Added press event to queue")
        except Exception as e:
            self._logger.error(f"Error in key press handler: {e}")

    def _on_key_release(self, key):
        """Handle key release events - synchronous callback"""
        try:
            if key == keyboard.Key.space:
                self._logger.debug(f"Space key released, current state: pressed={self._pressed}")
                if self._pressed:
                    self._pressed = False
                    self._key_queue.put(('release', None))
                    self._logger.debug("Added release event to queue")
        except Exception as e:
            self._logger.error(f"Error in key release handler: {e}")

    async def _process_keyboard_events(self):
        """Process keyboard events from the queue"""
        self._logger.info("Starting keyboard event processing loop")
        while True:
            try:
                # Use get_nowait to avoid blocking
                try:
                    event_type, _ = self._key_queue.get_nowait()
                    self._logger.debug(f"Processing keyboard event: {event_type}")
                    if event_type == 'press':
                        await self._start_recording()
                    elif event_type == 'release':
                        await self._stop_recording()
                except queue.Empty:
                    pass
                # Yield to other tasks
                await asyncio.sleep(0.01)  # Reduced sleep time for more responsive handling
            except Exception as e:
                self._logger.error(f"Error processing keyboard event: {e}")
                await asyncio.sleep(0.1)  # Longer sleep on error

    async def _start_recording(self):
        """Start recording audio"""
        try:
            self._logger.info("Starting recording")
            self.recorder.start_recording()  # This is synchronous
            await self.event_bus.publish(Event(
                type=EventType.RECORDING_START,
                source='AudioListener',
                content="Recording started",
                timestamp=time.time()
            ))
        except Exception as e:
            self._logger.error(f"Error starting recording: {e}")
            await self.event_bus.publish(Event(
                type=EventType.ERROR,
                source='AudioListener',
                content=f"Failed to start recording: {e}",
                timestamp=time.time()
            ))

    async def _stop_recording(self):
        """Stop recording and transcribe"""
        try:
            self._logger.info("Stopping recording")
            audio_file = self.recorder.stop_recording()  # This is synchronous
            if not audio_file:
                self._logger.warning("No audio file was recorded")
                await self.event_bus.publish(Event(
                    type=EventType.STATUS_UPDATE,
                    source='AudioListener',
                    content="ready",
                    timestamp=time.time()
                ))
                return

            await self.event_bus.publish(Event(
                type=EventType.RECORDING_STOP,
                source='AudioListener',
                content="Recording stopped",
                timestamp=time.time()
            ))

            self._logger.info(f"Transcribing audio file: {audio_file}")
            try:
                with open(audio_file, "rb") as f:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.model_name,
                        file=f
                    )
                await self.event_bus.publish(Event(
                    type=EventType.TRANSCRIPTION,
                    source='AudioListener',
                    content=transcript.text,
                    timestamp=time.time()
                ))
            except Exception as e:
                self._logger.error(f"Error in transcription: {e}")
                await self.event_bus.publish(Event(
                    type=EventType.ERROR,
                    source='AudioListener',
                    content=f"Error in transcription: {e}",
                    timestamp=time.time()
                ))
        except Exception as e:
            self._logger.error(f"Error stopping recording: {e}")
            await self.event_bus.publish(Event(
                type=EventType.ERROR,
                source='AudioListener',
                content=f"Failed to stop recording: {e}",
                timestamp=time.time()
            ))

    async def start(self):
        """Start the audio listener"""
        # Store the event loop
        self._loop = asyncio.get_running_loop()
        
        # Start keyboard listener in a separate thread
        def start_keyboard_listener():
            try:
                self._logger.info("Starting keyboard listener")
                self._keyboard_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                    suppress=False  # Don't suppress other keyboard events
                )
                self._keyboard_listener.start()
                self._logger.info("Keyboard listener started successfully")
            except Exception as e:
                self._logger.error(f"Failed to start keyboard listener: {e}")
                raise

        # Start keyboard listener in a separate thread
        keyboard_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        # Start processing keyboard events
        asyncio.create_task(self._process_keyboard_events())
        self._logger.info("Audio listener started")

    async def stop(self):
        """Stop the audio listener"""
        self._logger.info("Stopping audio listener")
        if self._keyboard_listener:
            try:
                self._keyboard_listener.stop()
                self._logger.info("Keyboard listener stopped")
            except Exception as e:
                self._logger.error(f"Error stopping keyboard listener: {e}")
        self.recorder.cleanup()
        self._logger.info("Audio listener stopped")

class AOIKeyboardListener(AOIModule):
    """Handles keyboard text input"""
    def __init__(self, event_bus: EventBus):
        event_types_to_subscribe = {
            EventType.STATUS_UPDATE
        }
        super().__init__("KeyboardListener", event_bus, event_types_to_subscribe)
        self._running = False
        self._input_queue = queue.Queue()
        self._input_thread = None

    def _input_loop(self):
        """Synchronous input loop running in separate thread"""
        try:
            print("\n=== Keyboard Input Mode ===")
            print("Type your message and press Enter to send.")
            print("Type 'quit' to exit.\n")
            
            while self._running:
                try:
                    # Get input from user
                    user_input = input("User: ").strip()
                    
                    if user_input.lower() == 'quit':
                        self._running = False
                        break
                    
                    if user_input:
                        # Put input in queue for async processing
                        self._input_queue.put(user_input)
                        
                except EOFError:
                    break
                except Exception as e:
                    self._logger.error(f"Error in keyboard input: {e}")
                    break
        except Exception as e:
            self._logger.error(f"Error in input loop: {e}")

    async def run(self):
        """Main event processing loop"""
        self._logger.info("KeyboardListener started")
        self._running = True
        
        # Start input thread
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()
        
        # Process input from queue
        while self._running:
            try:
                # Check for input with timeout
                try:
                    user_input = self._input_queue.get_nowait()
                    
                    # Emit keyboard input event (treat as transcription for AI processing)
                    await self.event_bus.publish(Event(
                        type=EventType.TRANSCRIPTION,
                        source='KeyboardListener',
                        content=user_input,
                        timestamp=time.time()
                    ))
                    
                except queue.Empty:
                    pass
                
                # Yield to other tasks
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self._logger.error(f"Error in keyboard listener: {e}")
                await asyncio.sleep(0.1)

    async def stop(self):
        """Stop the keyboard listener"""
        self._running = False
        if self._input_thread:
            self._input_thread.join(timeout=1.0)
        self._logger.info("KeyboardListener stopped")

class AOIConsoleUI(AOIModule):
    """Handles console input/output with separate terminal for AI responses"""
    def __init__(self, event_bus: EventBus):
        event_types_to_subscribe = {
            EventType.RECORDING_START,
            EventType.RECORDING_STOP,
            EventType.TRANSCRIPTION,
            EventType.LLM_RESPONSE,
            EventType.ERROR,
            EventType.STATUS_UPDATE,
            EventType.TTS_START,
            EventType.TTS_COMPLETE
        }
        super().__init__("ConsoleUI", event_bus, event_types_to_subscribe)
        self._waiting_for_response = False
        self._last_event_time = 0
        self._initialized = False
        self._speaking = False
        self._response_terminal = None
        self._response_file = None

    def _open_response_terminal(self):
        """Open a separate terminal window for AI responses"""
        try:
            # Create a temporary file to write responses
            self._response_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            
            # Platform-specific terminal opening
            if platform.system() == 'Darwin':  # macOS
                # Use Terminal.app with a new window
                subprocess.Popen([
                    'osascript', '-e', 
                    f'tell application "Terminal" to do script "tail -f {self._response_file.name}"'
                ])
            elif platform.system() == 'Linux':
                # Use xterm or gnome-terminal
                try:
                    subprocess.Popen(['gnome-terminal', '--', 'tail', '-f', self._response_file.name])
                except FileNotFoundError:
                    subprocess.Popen(['xterm', '-e', f'tail -f {self._response_file.name}'])
            elif platform.system() == 'Windows':
                # Use cmd with start
                subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', f'type {self._response_file.name}'])
            
            self._logger.info(f"Opened response terminal with file: {self._response_file.name}")
            
        except Exception as e:
            self._logger.error(f"Failed to open response terminal: {e}")
            # Fallback to regular console output
            self._response_file = None

    def _write_to_response_terminal(self, message: str):
        """Write message to the response terminal"""
        try:
            if self._response_file:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                self._response_file.write(f"[{timestamp}] {message}\n")
                self._response_file.flush()
            else:
                # Fallback to regular console
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
        except Exception as e:
            self._logger.error(f"Error writing to response terminal: {e}")
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")  # Fallback

    async def run(self):
        """Main event processing loop"""
        self._logger.info("ConsoleUI started")
        
        # Open separate terminal for responses
        self._open_response_terminal()
        
        # Print welcome message only once
        if not self._initialized:
            print("\n=== Welcome to Async-AOI ===")
            print("A minimal AI system that listens and responds to your voice.")
            print("Press and hold SPACE to start recording, release to stop.")
            print("AI responses will appear in a separate terminal window.")
            print("Press Ctrl+C to exit.\n")
            self._initialized = True

        while True:
            try:
                event = await self._queue.get()
                await self._handle_event(event)
            except Exception as e:
                self._logger.error(f"Error in ConsoleUI: {e}")

    async def _handle_event(self, event: Event):
        """Process and display events"""
        current_time = event.timestamp
        if current_time - self._last_event_time < 0.1:
            return
        self._last_event_time = current_time

        if event.type == EventType.RECORDING_START:
            self._write_to_response_terminal("Recording... Release space to stop")
            self._waiting_for_response = False
            self._speaking = False
        elif event.type == EventType.RECORDING_STOP:
            self._write_to_response_terminal("Transcribing...")
            self._waiting_for_response = True
            self._speaking = False
        elif event.type == EventType.TRANSCRIPTION:
            self._write_to_response_terminal(f"You said: {event.content}")
        elif event.type == EventType.LLM_RESPONSE:
            self._write_to_response_terminal(f"AI: {event.content}")
            self._waiting_for_response = True
        elif event.type == EventType.ERROR:
            self._write_to_response_terminal(f"Error: {event.content}")
            self._waiting_for_response = True
        elif event.type == EventType.STATUS_UPDATE:
            if event.content == "ready":
                self._waiting_for_response = False
                self._speaking = False
                self._write_to_response_terminal("Press space to start speaking")
        elif event.type == EventType.TTS_START:
            self._speaking = True
            self._write_to_response_terminal("Speaking...")
        elif event.type == EventType.TTS_COMPLETE:
            self._speaking = False
            if not self._waiting_for_response:
                self._write_to_response_terminal("Press space to start speaking")

    async def stop(self):
        """Stop the console UI and cleanup"""
        if self._response_file:
            try:
                self._response_file.close()
                os.unlink(self._response_file.name)
            except Exception as e:
                self._logger.error(f"Error cleaning up response file: {e}")
        self._logger.info("ConsoleUI stopped")

class AOISpeaker(AOIModule):
    """Handles text-to-speech using OpenAI's TTS model"""
    def __init__(self, event_bus: EventBus, model_name: str = "tts-1"):
        event_types_to_subscribe = {
            EventType.LLM_RESPONSE,
            EventType.ERROR,
            EventType.STATUS_UPDATE
        }
        super().__init__("Speaker", event_bus, event_types_to_subscribe)
        self.model_name = model_name
        self._init_model()
        self._welcome_said = False
        self._speaking = False

    def _init_model(self):
        """Initialize OpenAI client"""
        try:
            with open(OPENAI_API_KEY_FILE, 'r') as f:
                api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key file is empty")
            self.client = openai.OpenAI(api_key=api_key)
            self._logger.info("OpenAI client initialized for TTS")
        except Exception as e:
            self._logger.error(f"Error initializing OpenAI client for TTS: {e}")
            raise

    async def _speak(self, text: str):
        """Convert text to speech and play it"""
        if self._speaking:
            self._logger.warning("Already speaking, ignoring new speech request")
            return

        try:
            self._speaking = True
            self._logger.info(f"Converting to speech: {text[:100]}...")
            await self.event_bus.emit(Event(
                type=EventType.TTS_START,
                content="Speaking...",
                timestamp=time.time(),
                source="Speaker"
            ))

            # Generate speech using non-streaming API
            response = self.client.audio.speech.create(
                model=self.model_name,
                voice="nova",
                input=text
            )

            # Save to temporary file
            temp_file = f"temp_speech_{int(time.time())}.mp3"
            response.stream_to_file(temp_file)

            # Play the audio
            try:
                if platform.system() == 'Darwin':  # macOS
                    os.system(f'afplay {temp_file}')
                elif platform.system() == 'Linux':
                    os.system(f'mpg123 {temp_file}')
                elif platform.system() == 'Windows':
                    os.system(f'start {temp_file}')
                else:
                    self._logger.warning(f"Unsupported platform for audio playback: {platform.system()}")
            finally:
                # Clean up the temporary file
                try:
                    os.remove(temp_file)
                except Exception as e:
                    self._logger.error(f"Error removing temporary file: {e}")

            await self.event_bus.emit(Event(
                type=EventType.TTS_COMPLETE,
                content="Speech complete",
                timestamp=time.time(),
                source="Speaker"
            ))

        except Exception as e:
            self._logger.error(f"Error in text-to-speech: {e}")
            await self.event_bus.emit(Event(
                type=EventType.ERROR,
                content=f"Text-to-speech error: {e}",
                timestamp=time.time(),
                source="Speaker"
            ))
        finally:
            self._speaking = False
            # Always emit ready status after speaking is done
            await self.event_bus.emit(Event(
                type=EventType.STATUS_UPDATE,
                content="ready",
                timestamp=time.time(),
                source="Speaker"
            ))

    async def run(self):
        """Main event processing loop"""
        self._logger.info("Speaker started")
        
        # Say welcome message only once at startup
        if not self._welcome_said:
            if LANGUAGE == "EN":
                await self._speak("How can I help you today?")
            elif LANGUAGE == "ZH":
                await self._speak("你好，有什么可以帮你的吗？")
            self._welcome_said = True

        while True:
            try:
                event = await self._queue.get()
                if event.type == EventType.LLM_RESPONSE and not self._speaking:
                    await self._speak(event.content)
                elif event.type == EventType.ERROR and not self._speaking:
                    await self._speak(f"I encountered an error: {event.content}")
            except Exception as e:
                self._logger.error(f"Error in Speaker: {e}")

""" ===================== AICore ===================== """
class AOICore(AOIModule):
    """Handles LLM interactions using dialogue engine"""
    def __init__(self, event_bus: EventBus, model_name: str = "gpt-4o-mini"):
        event_types_to_subscribe = {
            EventType.TRANSCRIPTION
        }
        super().__init__("AOICore", event_bus, event_types_to_subscribe)
        self.model_name = model_name
        self._init_model()
        
    def _init_model(self):
        """Initialize dialogue engine"""
        try:
            # Initialize dialogue engine
            save_dir = "outputs/0618/dev1"
            os.makedirs(save_dir, exist_ok=True)
            self.dialogue_engine = AOIDialogueEngine(
                model_name=self.model_name,
                save_dir=save_dir,
                api_key_file=OPENAI_API_KEY_FILE,
                locale=LANGUAGE
            )
            self.dialogue_engine.start_new_session()
            self._logger.info("Dialogue engine initialized")
        except Exception as e:
            self._logger.error(f"Error initializing dialogue engine: {e}")
            raise

    async def _handle_transcription(self, event: Event):
        """Process transcription and generate response using dialogue engine"""
        try:
            self._logger.info(f"Generating response for: {event.content[:100]}")
            
            # Use dialogue engine to process the turn
            response = self.dialogue_engine.run_turn(event.content)
            
            await self.event_bus.publish(Event(
                type=EventType.LLM_RESPONSE,
                source='AOICore',
                content=response
            ))
            self._logger.info("Response generated successfully")
            
        except Exception as e:
            self._logger.error(f"Error generating response: {e}")
            await self.event_bus.publish(Event(
                type=EventType.ERROR,
                source='AOICore',
                content=str(e)
            ))

    async def run(self):
        """Main event processing loop"""
        self._logger.info("AOICore started")
        while True:
            try:
                event = await self._queue.get()
                if event.type == EventType.TRANSCRIPTION:
                    await self._handle_transcription(event)
            except Exception as e:
                self._logger.error(f"Error in AOICore: {e}")

""""===================== Main ===================== """

async def main():
    """Main entry point"""
    try:
        # Initialize components
        event_bus = EventBus()
        ai_responder = AOICore(event_bus)
        console_ui = AOIConsoleUI(event_bus)

        # Start components
        responder_task = asyncio.create_task(ai_responder.run())
        ui_task = asyncio.create_task(console_ui.run())

        tasks_to_gather = [responder_task, ui_task]

        if INPUT_MODE == "voice":
            audio_listener = AOIAudioListener(event_bus)
            await audio_listener.start()
        elif INPUT_MODE == "keyboard":
            keyboard_listener = AOIKeyboardListener(event_bus)
            keyboard_task = asyncio.create_task(keyboard_listener.run())
            tasks_to_gather.append(keyboard_task)
        else:
            raise ValueError(f"Invalid input mode: {INPUT_MODE}")
        
        if OUTPUT_MODE == "speaker":
            speaker = AOISpeaker(event_bus)
            speaker_task = asyncio.create_task(speaker.run())
            tasks_to_gather.append(speaker_task)

        # Wait for tasks
        await asyncio.gather(*tasks_to_gather)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await keyboard_listener.stop()
        if INPUT_MODE == "voice":
            await audio_listener.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...") 