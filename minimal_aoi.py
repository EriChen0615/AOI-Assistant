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
from typing import Dict, Set, Any
import openai

@dataclass
class AOIEvent:
    meta_data: dict
    content: str

class Memory:
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._permissions: Dict[str, Set[str]] = {
            "Listener": {"read": {"user_input"}, "write": {"user_input"}},
            "Displayer": {"read": {"llm_output"}, "write": {"llm_output"}},
            "Controller": {"read": {"user_input"}, "write": {"llm_output", "user_input"}}
        }
    
    def read(self, module_name: str, key: str) -> Any:
        with self._lock:
            if key not in self._permissions[module_name]["read"]:
                raise PermissionError(f"Module {module_name} not allowed to read {key}")
            return self._data.get(key)
    
    def write(self, module_name: str, key: str, value: Any):
        with self._lock:
            if key not in self._permissions[module_name]["write"]:
                raise PermissionError(f"Module {module_name} not allowed to write {key}")
            self._data[key] = value

class AOIModule(ABC):
    def __init__(self, name: str, memory: Memory, event_queue: queue.Queue):
        self.name = name
        self.memory = memory
        self.event_queue = event_queue
        self._thread = None
        self._stop_event = threading.Event()
    
    def emit_event(self, event: AOIEvent):
        self.event_queue.put(event)
    
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def run(self):
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
            self.run()

class AOIListener(AOIModule):
    def __init__(self, memory: Memory, event_queue: queue.Queue):
        super().__init__("Listener", memory, event_queue)

    def initialize(self):
        pass

    def run(self):
        try:
            user_input = input("User: ")
            self.memory.write(self.name, "user_input", user_input)
            self.emit_event(AOIEvent(
                meta_data={"sent_from": self.name, "type": "user_input"},
                content=user_input
            ))
        except EOFError:
            self._stop_event.set()

class AOIDisplayer(AOIModule):
    def __init__(self, memory: Memory, event_queue: queue.Queue):
        super().__init__("Displayer", memory, event_queue)

    def initialize(self):
        pass

    def run(self):
        output = self.memory.read(self.name, "llm_output")
        if output:
            print(f"AI: {output}")
            self.memory.write(self.name, "llm_output", None)  # Clear after display

class Dispatcher:
    def __init__(self):
        self.memory = Memory()
        self.event_queue = queue.Queue()
        self.modules = {}
        self._stop_event = threading.Event()
        self._thread = None
    
    def register_module(self, module: AOIModule):
        self.modules[module.name] = module
    
    def on_event(self, event: AOIEvent):
        event_meta = event.meta_data
        if event_meta["sent_from"] == "Listener":
            if event_meta["type"] == "user_input":
                self.modules["Controller"].emit_event(event)
        elif event_meta["sent_from"] == "Controller":
            if "llm_output" in event.meta_data:
                self.memory.write("Controller", "llm_output", event.content)
                self.modules["Displayer"].emit_event(event)
    
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
            try:
                event = self.event_queue.get(timeout=1.0)
                self.on_event(event)
            except queue.Empty:
                continue

class Controller(AOIModule):
    def __init__(self, memory: Memory, event_queue: queue.Queue, model_name: str = "gpt-3.5-turbo"):
        super().__init__("Controller", memory, event_queue)
        self.model_name = model_name
        self._init_model(model_name)
    
    def _init_model(self, model_name: str):
        if model_name in ['gpt-3.5-turbo']:
            self.client = openai.OpenAI()  # You'll need to set OPENAI_API_KEY
        else:
            raise NotImplementedError(f"Model name = {model_name} is not supported!")
    
    def emit_event(self, event: AOIEvent):
        super().emit_event(event)
    
    def initialize(self):
        pass
    
    def run(self):
        user_input = self.memory.read(self.name, "user_input")
        if user_input:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": user_input}]
                )
                output = response.choices[0].message.content
                self.memory.write(self.name, "llm_output", output)
                self.emit_event(AOIEvent(
                    meta_data={"sent_from": self.name, "type": "llm_output"},
                    content=output
                ))
                # Clear the input after processing
                self.memory.write(self.name, "user_input", None)
            except Exception as e:
                print(f"Error in Controller: {e}")

def main():
    dispatcher = Dispatcher()
    
    # Create and register modules
    listener = AOIListener(dispatcher.memory, dispatcher.event_queue)
    displayer = AOIDisplayer(dispatcher.memory, dispatcher.event_queue)
    controller = Controller(dispatcher.memory, dispatcher.event_queue)
    
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