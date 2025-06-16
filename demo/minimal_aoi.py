"""
Minimal Artificial Open Intelligence. Min-AOI.

Min-AOI has four components:

1. Listener: runs in its own thread. Take console input to memory upon user input.
2. Displayer: runs in its own thread. Print output to console once it's available.
3. (Rule-based) Dispatcher: keeps track of memory. When a module generates an event, the dispatcher decides whether it needs to channel the change to other modules or the Controller.
4. (LLM-driven) Controller: given the memory and context, generate action
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import queue
from typing import Dict, Set, Any, Callable
import openai
import pyaudio
import wave
from pynput import keyboard
import tempfile
import os
import platform
import sys
import logging
import functools
import datetime
import json
import copy

# Set up logging
logging.basicConfig(
    filename='aoi_events.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def log_event(func: Callable) -> Callable:
    """Decorator to log events in the system"""
    @functools.wraps(func)
    def wrapper(self, event: AOIEvent, *args, **kwargs):
        # Log the event before handling
        logging.info(f"Event received in {self.name}: {json.dumps(event.meta_data)} - Content: {event.content[:100]}")
        try:
            result = func(self, event, *args, **kwargs)
            logging.info(f"Event handled successfully in {self.name}")
            return result
        except Exception as e:
            logging.error(f"Error handling event in {self.name}: {str(e)}")
            raise
    return wrapper

def log_emit(func: Callable) -> Callable:
    """Decorator to log event emissions"""
    @functools.wraps(func)
    def wrapper(self, event: AOIEvent, *args, **kwargs):
        logging.info(f"Event emitted from {self.name}: {json.dumps(event.meta_data)} - Content: {event.content[:100]}")
        return func(self, event, *args, **kwargs)
    return wrapper

@dataclass
class AOIEvent:
    meta_data: dict
    content: str

class Memory:
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        # Remove module-specific permissions since only Dispatcher will access memory
        self._permissions = {"Dispatcher": {"read": {"user_input", "llm_output", "recording_status", "transcription"}, 
                                          "write": {"user_input", "llm_output", "recording_status", "transcription"}}}
    
    def read(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key)
    
    def write(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value

class AudioRecorder:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.recording = False
        self.frames = []
        self.p = None
        self.stream = None
        self.last_audio_file = None  # Keep track of the last recorded file

    def start_recording(self):
        if self.recording:
            return
            
        self.recording = True
        self.frames = []
        logging.info("Starting audio recording")
        
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=None  # Use default input device
            )
            
            def record():
                while self.recording and self.stream:
                    try:
                        data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                        self.frames.append(data)
                    except Exception as e:
                        logging.error(f"Error reading audio: {e}")
                        break
            
            self.record_thread = threading.Thread(target=record)
            self.record_thread.daemon = True
            self.record_thread.start()
            logging.info("Audio recording thread started")
            
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            self.cleanup()
            raise

    def stop_recording(self):
        if not self.recording:
            logging.warning("Attempted to stop recording when not recording")
            return None
            
        self.recording = False
        logging.info("Stopping audio recording")
        
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
        
        if not self.frames:
            logging.warning("No audio frames recorded")
            return None
            
        try:
            # Create a timestamped filename in the current directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            
            # Save the recording
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            self.last_audio_file = filename
            logging.info(f"Audio saved to {filename}")
            print(f"\nAudio saved to {filename} for inspection")
            return filename
            
        except Exception as e:
            logging.error(f"Error saving recording: {e}")
            return None
        finally:
            self.cleanup()

    def cleanup(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
            
        if self.p:
            try:
                self.p.terminate()
            except:
                pass
            self.p = None
            
        self.recording = False
        self.frames = []

class AOIModule(ABC):
    def __init__(self, name: str, input_queue: queue.Queue, output_queue: queue.Queue):
        self.name = name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self._thread = None
        self._stop_event = threading.Event()
    
    @log_emit
    def emit_event(self, event: AOIEvent):
        self.output_queue.put(event)
    
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def handle_event(self, event: AOIEvent):
        """Handle a single event from the input queue"""
        pass
    
    def start(self):
        self._thread = threading.Thread(target=self._run_wrapper)
        self._thread.start()
    
    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
    
    def _run_wrapper(self):
        self.initialize()
        while not self._stop_event.is_set():
            try:
                event = self.input_queue.get(timeout=1.0)
                print(f"Module {self.name} got event: {event.meta_data['type']}")
                self.handle_event(event)
            except queue.Empty:
                continue

class AOIListener(AOIModule):
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue, model_name: str = "gpt-4o-transcribe"):
        super().__init__("Listener", input_queue, output_queue)
        self.recorder = AudioRecorder()
        self._keyboard_listener = None
        self.model_name = model_name
        self._init_model()
        self._pressed = False
        # Initial status will be set by Dispatcher
    
    def _init_model(self):
        if self.model_name in ['gpt-4o-transcribe']:
            try:
                with open('configs/openai_api_key', 'r') as f:
                    api_key = f.read().strip()
                if not api_key:
                    raise ValueError("API key file is empty")
                print("Successfully loaded API key")
                self.client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                print(f"Error initializing OpenAI client: {str(e)}")
                raise
        else:
            raise NotImplementedError(f"Model name = {self.model_name} is not supported!")
    

    def initialize(self):
        # Try to start the listener anyway, it might work if permissions are already granted
        try:
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self._keyboard_listener.start()
        except Exception as e:
            print(f"Error starting keyboard listener: {e}")
            print("Please grant accessibility permissions and restart the application.")
            if platform.system() == 'Darwin':
                print("Please follow these steps:")
                print("1. Open System Preferences")
                print("2. Go to Security & Privacy > Privacy > Accessibility")
                print("3. Click the lock icon to make changes (you'll need to enter your password)")
                print("4. Add Terminal (or your IDE) to the list of allowed apps")
                print("5. Restart the application\n")
                sys.exit(1)
    
    @log_event
    def handle_event(self, event: AOIEvent):
        pass

    def _on_key_press(self, key):
        if key == keyboard.Key.space:
            if not self._pressed:
                self.emit_event(AOIEvent(
                    meta_data={"sent_from": self.name, "type": "recording_start"},
                    content=""
                ))
                self._pressed = True
                self.recorder.start_recording()
            else:
                pass

    def _on_key_release(self, key):
        if key == keyboard.Key.space:
            self.emit_event(AOIEvent(
                meta_data={"sent_from": self.name, "type": "recording_stop"},
                content=""
            ))
            
            audio_file = self.recorder.stop_recording()
            if not audio_file:
                logging.warning("No audio file was recorded")
                self.emit_event(AOIEvent(
                    meta_data={"sent_from": self.name, "type": "status_update"},
                    content="ready"
                ))
                return
            
            try:
                print(f"\nTranscribing audio file: {audio_file}")
                with open(audio_file, "rb") as f:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.model_name,
                        file=f
                    )
                
                # Emit transcription event
                self.emit_event(AOIEvent(
                    meta_data={"sent_from": self.name, "type": "transcription"},
                    content=transcript.text
                ))
            except Exception as e:
                logging.error(f"Error in transcription: {e}")
                print(f"\nError in transcription: {e}")
                self.emit_event(AOIEvent(
                    meta_data={"sent_from": self.name, "type": "status_update"},
                    content="ready"
                ))

    def stop(self):
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        self.recorder.cleanup()
        super().stop()

class AOIDisplayer(AOIModule):
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue):
        super().__init__("Displayer", input_queue, output_queue)
        self._waiting_for_response = False

    def initialize(self):
        print("\n=== Welcome to Min-AOI ===")
        print("A minimal AI system that listens and responds to your voice.")
        print("Press and hold SPACE to start recording, release to stop.")
        print("Press Ctrl+C to exit.\n")
        print("Press space to start speaking", end="", flush=True)

    @log_event
    def handle_event(self, event: AOIEvent):
        event_meta = event.meta_data
        if event_meta["type"] == "recording_start":
            print("\rRecording... Release space to stop", end="", flush=True)
            self._waiting_for_response = False
        elif event_meta["type"] == "recording_stop":
            print("\rTranscribing...", end="", flush=True)
            self._waiting_for_response = True
        elif event_meta["type"] == "transcription":
            print(f"\nYou said: {event.content}")
        elif event_meta["type"] == "llm_output":
            print(f"\nAI: {event.content}")
            print("\rPress space to start speaking", end="", flush=True)
            self._waiting_for_response = False
        elif event_meta["type"] == "status_update":
            if event.content == "ready" and not self._waiting_for_response:
                print("\rPress space to start speaking", end="", flush=True)

class Controller(AOIModule):
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue, model_name: str = "gpt-3.5-turbo"):
        super().__init__("Controller", input_queue, output_queue)
        self.model_name = model_name
        self._init_model(model_name)
    
    def _init_model(self, model_name: str):
        if model_name in ['gpt-3.5-turbo']:
            try:
                with open('configs/openai_api_key', 'r') as f:
                    api_key = f.read().strip()
                if not api_key:
                    raise ValueError("API key file is empty")
                print("Successfully loaded API key")
                self.client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                print(f"Error initializing OpenAI client: {str(e)}")
                raise
        else:
            raise NotImplementedError(f"Model name = {model_name} is not supported!")
    
    def initialize(self):
        logging.info("Controller initialized")
    
    @log_event
    def handle_event(self, event: AOIEvent):
        if event.meta_data["type"] == "transcription":
            logging.info(f"Controller received transcription: {event.content[:100]}")
            try:
                # print("\nGenerating response...", end="", flush=True)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": event.content}]
                )
                output = response.choices[0].message.content
                logging.info(f"Controller generated response: {output[:100]}")
                # print("\rResponse generated!", end="", flush=True)
                
                # Emit LLM output event
                self.emit_event(AOIEvent(
                    meta_data={"sent_from": self.name, "type": "llm_output"},
                    content=output
                ))
            except Exception as e:
                error_msg = f"Error in Controller: {e}"
                logging.error(error_msg)
                print(f"\n{error_msg}")
                # Emit error event to displayer
                self.emit_event(AOIEvent(
                    meta_data={"sent_from": self.name, "type": "error"},
                    content=error_msg
                ))

class Dispatcher:
    def __init__(self):
        self.memory = Memory()
        # Create queues for each module's input and output
        self.listener_input_queue = queue.Queue()
        self.listener_output_queue = queue.Queue()
        self.displayer_input_queue = queue.Queue()
        self.displayer_output_queue = queue.Queue()
        self.controller_input_queue = queue.Queue()
        self.controller_output_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self.modules = {}
        
        # Initialize memory state
        self.memory.write("recording_status", "ready")
    
    def register_module(self, module: AOIModule):
        self.modules[module.name] = module
    
    def start(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        
        # Start all modules
        for module in self.modules.values():
            module.start()
    
    def stop(self):
        self._stop_event.set()
        for module in self.modules.values():
            module.stop()
        if self._thread:
            self._thread.join()
    
    def _run(self):
        while not self._stop_event.is_set():
            # Check all module output queues
            queues_to_check = {
                "Listener": self.listener_output_queue,
                "Displayer": self.displayer_output_queue,
                "Controller": self.controller_output_queue
            }
            
            for module_name, output_queue in queues_to_check.items():
                try:
                    event = output_queue.get_nowait()
                    # Process the event
                    self._handle_event(copy.deepcopy(event))
                    # Now remove it from the queue
                    output_queue.get()  # Remove the event we just processed
                except queue.Empty:
                    continue
    
    def _handle_event(self, event: AOIEvent):
        event_meta = event.meta_data
        if event_meta["sent_from"] == "Listener":
            if event_meta["type"] == "recording_start":
                self.memory.write("recording_status", "recording")
                self.displayer_input_queue.put(event)
            elif event_meta["type"] == "recording_stop":
                self.memory.write("recording_status", "transcribing")
                self.displayer_input_queue.put(event)
            elif event_meta["type"] == "transcription":
                self.memory.write("transcription", event.content)
                self.memory.write("user_input", event.content)
                self.displayer_input_queue.put(event)
                self.controller_input_queue.put(event)
            elif event_meta["type"] == "status_update":
                self.memory.write("recording_status", event.content)
                self.displayer_input_queue.put(event)
            else:
                logging.warning(f"Unknown event type from Listener: {event_meta['type']}")
        elif event_meta["sent_from"] == "Controller":
            if event_meta["type"] == "llm_output":
                self.memory.write("llm_output", event.content)
                self.displayer_input_queue.put(event)
                # Clear user input after processing
                self.memory.write("user_input", None)
                # Send ready status
                self.displayer_input_queue.put(AOIEvent(
                    meta_data={"sent_from": "Dispatcher", "type": "status_update"},
                    content="ready"
                ))
            elif event_meta["type"] == "error":
                self.displayer_input_queue.put(event)
                # Send ready status after error
                self.displayer_input_queue.put(AOIEvent(
                    meta_data={"sent_from": "Dispatcher", "type": "status_update"},
                    content="ready"
                ))
            else:
                logging.warning(f"Unknown event type from Controller: {event_meta['type']}")
        else:
            logging.warning(f"Unknown event source: {event_meta['sent_from']}")

def main():
    dispatcher = Dispatcher()
    
    # Create modules with their respective input and output queues
    listener = AOIListener(dispatcher.listener_input_queue, dispatcher.listener_output_queue)
    displayer = AOIDisplayer(dispatcher.displayer_input_queue, dispatcher.displayer_output_queue)
    controller = Controller(dispatcher.controller_input_queue, dispatcher.controller_output_queue)
    
    dispatcher.register_module(listener)
    dispatcher.register_module(displayer)
    dispatcher.register_module(controller)
    
    try:
        dispatcher.start()
        # Keep main thread alive
        while True:
            threading.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        dispatcher.stop()

if __name__ == "__main__":
    main()