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
DEBUG_LOGGING = False

import asyncio
import logging
import json
import datetime
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Set
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

# Set up logging with configurable debug level
if DEBUG_LOGGING:
    log_level = logging.DEBUG
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
    
    logger.info("Async-AOI starting up...")
    logger.debug("Debug logging enabled")
else:
    # Production mode: disable all logging
    logging.disable(logging.CRITICAL)
    logger = logging.getLogger()
    logger.disabled = True

# Create a silent logger for production mode
class SilentLogger:
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
    def critical(self, msg): pass

# Use silent logger in production mode
if not DEBUG_LOGGING:
    logger = SilentLogger()

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

class AudioListener:
    """Handles audio recording and transcription"""
    def __init__(self, event_bus: EventBus, model_name: str = "whisper-1"):
        self.event_bus = event_bus
        self.model_name = model_name
        self.recorder = AudioRecorder()
        self._keyboard_listener = None
        self._pressed = False
        self._logger = logging.getLogger('AudioListener')
        self._key_queue = queue.Queue()  # Use synchronous queue for keyboard events
        self._init_model()
        self._queue = event_bus.register_queue('AudioListener')
        self._loop = None  # Store the event loop
        event_bus.subscribe('AudioListener', {
            EventType.STATUS_UPDATE
        })

    def _init_model(self):
        """Initialize OpenAI client"""
        try:
            with open('configs/openai_api_key', 'r') as f:
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

class AIResponder:
    """Handles LLM interactions"""
    def __init__(self, event_bus: EventBus, model_name: str = "gpt-3.5-turbo"):
        self.event_bus = event_bus
        self.model_name = model_name
        self._logger = logging.getLogger('AIResponder')
        self._queue = event_bus.register_queue('AIResponder')
        self._init_model()
        self._conversation_history = []  # Store conversation history
        self._max_history = 10  # Maximum number of turns to keep
        
        # Add system message to define AI identity
        self._system_message = {
            "role": "system", 
            "content": "You are AOI (pronounced as ah-o-e), a LLM-driven personal assistant built by Jinghong Chen."
        }
        
        event_bus.subscribe('AIResponder', {
            EventType.TRANSCRIPTION
        })

    def _init_model(self):
        """Initialize OpenAI client"""
        try:
            with open('configs/openai_api_key', 'r') as f:
                api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key file is empty")
            self.client = openai.OpenAI(api_key=api_key)
            self._logger.info("OpenAI client initialized")
        except Exception as e:
            self._logger.error(f"Error initializing OpenAI client: {e}")
            raise

    async def _handle_transcription(self, event: Event):
        """Process transcription and generate response"""
        try:
            self._logger.info(f"Generating response for: {event.content[:100]}")
            
            # Add user message to history
            self._conversation_history.append({"role": "user", "content": event.content})
            
            # Trim history if it exceeds max length
            if len(self._conversation_history) > self._max_history:
                self._conversation_history = self._conversation_history[-self._max_history:]
            
            # Prepare messages with system message and conversation history
            messages = [self._system_message] + self._conversation_history
            
            # Generate response using conversation history
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            output = response.choices[0].message.content
            
            # Add assistant response to history
            self._conversation_history.append({"role": "assistant", "content": output})
            
            await self.event_bus.publish(Event(
                type=EventType.LLM_RESPONSE,
                source='AIResponder',
                content=output
            ))
            self._logger.info("Response generated successfully")
            
        except Exception as e:
            self._logger.error(f"Error generating response: {e}")
            await self.event_bus.publish(Event(
                type=EventType.ERROR,
                source='AIResponder',
                content=str(e)
            ))

    async def run(self):
        """Main event processing loop"""
        self._logger.info("AIResponder started")
        while True:
            try:
                event = await self._queue.get()
                if event.type == EventType.TRANSCRIPTION:
                    await self._handle_transcription(event)
            except Exception as e:
                self._logger.error(f"Error in AIResponder: {e}")

class Speaker:
    """Handles text-to-speech using OpenAI's TTS model"""
    def __init__(self, event_bus: EventBus, model_name: str = "tts-1"):
        self.event_bus = event_bus
        self.model_name = model_name
        self._logger = logging.getLogger('Speaker')
        self._queue = event_bus.register_queue('Speaker')
        self._init_model()
        self._welcome_said = False
        self._speaking = False
        event_bus.subscribe('Speaker', {
            EventType.LLM_RESPONSE,
            EventType.ERROR,
            EventType.STATUS_UPDATE
        })

    def _init_model(self):
        """Initialize OpenAI client"""
        try:
            with open('configs/openai_api_key', 'r') as f:
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
            await self._speak("How can I help you today?")
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

class ConsoleUI:
    """Handles console input/output"""
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._logger = logging.getLogger('ConsoleUI')
        self._queue = event_bus.register_queue('ConsoleUI')
        self._waiting_for_response = False
        self._last_event_time = 0
        self._initialized = False
        self._speaking = False
        event_bus.subscribe('ConsoleUI', {
            EventType.RECORDING_START,
            EventType.RECORDING_STOP,
            EventType.TRANSCRIPTION,
            EventType.LLM_RESPONSE,
            EventType.ERROR,
            EventType.STATUS_UPDATE,
            EventType.TTS_START,
            EventType.TTS_COMPLETE
        })

    async def run(self):
        """Main event processing loop"""
        self._logger.info("ConsoleUI started")
        
        # Print welcome message only once
        if not self._initialized:
            print("\n=== Welcome to Async-AOI ===")
            print("A minimal AI system that listens and responds to your voice.")
            print("Press and hold SPACE to start recording, release to stop.")
            print("Press Ctrl+C to exit.\n")
            self._initialized = True
            # Don't print "Press space" here, wait for ready status

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
            print("\rRecording... Release space to stop", end="", flush=True)
            self._waiting_for_response = False
            self._speaking = False
        elif event.type == EventType.RECORDING_STOP:
            print("\rTranscribing...", end="", flush=True)
            self._waiting_for_response = True
            self._speaking = False
        elif event.type == EventType.TRANSCRIPTION:
            print(f"\nYou said: {event.content}")
        elif event.type == EventType.LLM_RESPONSE:
            print(f"\nAI: {event.content}")
            self._waiting_for_response = True
        elif event.type == EventType.ERROR:
            print(f"\nError: {event.content}")
            self._waiting_for_response = True
        elif event.type == EventType.STATUS_UPDATE:
            if event.content == "ready":
                self._waiting_for_response = False
                self._speaking = False
                print("\rPress space to start speaking", end="", flush=True)
        elif event.type == EventType.TTS_START:
            self._speaking = True
            print("\rSpeaking...", end="", flush=True)
        elif event.type == EventType.TTS_COMPLETE:
            self._speaking = False
            if not self._waiting_for_response:
                print("\rPress space to start speaking", end="", flush=True)

async def main():
    """Main entry point"""
    try:
        # Initialize components
        event_bus = EventBus()
        audio_listener = AudioListener(event_bus)
        ai_responder = AIResponder(event_bus)
        speaker = Speaker(event_bus)
        console_ui = ConsoleUI(event_bus)

        # Start components
        await audio_listener.start()
        responder_task = asyncio.create_task(ai_responder.run())
        speaker_task = asyncio.create_task(speaker.run())
        ui_task = asyncio.create_task(console_ui.run())

        # Wait for tasks
        await asyncio.gather(responder_task, speaker_task, ui_task)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await audio_listener.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...") 