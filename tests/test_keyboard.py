import unittest
import queue
import time
import threading
from pynput import keyboard
from demo.minimal_aoi import AOIListener, AOIEvent, AudioRecorder

class TestKeyboard(unittest.TestCase):
    def setUp(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.listener = AOIListener(self.input_queue, self.output_queue)
        # Initialize recorder properly
        self.listener.recorder = AudioRecorder()

    def tearDown(self):
        self.listener.stop()

    def test_space_key_events(self):
        """Test that space key press/release generates correct events"""
        print("\nPress and release the space key within 5 seconds...")
        self.listener.start()
        
        # Simulate space key press and release
        controller = keyboard.Controller()
        try:
            # Press space
            controller.press(keyboard.Key.space)
            time.sleep(0.1)  # Small delay
            
            # Check for recording_start event
            try:
                event = self.output_queue.get(timeout=1.0)
                self.assertEqual(event.meta_data["type"], "recording_start")
                print("Space press detected!")
            except queue.Empty:
                self.fail("Did not receive space press event")
            
            # Release space
            controller.release(keyboard.Key.space)
            time.sleep(0.1)  # Small delay
            
            # Check for recording_stop event
            try:
                event = self.output_queue.get(timeout=1.0)
                self.assertEqual(event.meta_data["type"], "recording_stop")
                print("Space release detected!")
            except queue.Empty:
                self.fail("Did not receive space release event")
        finally:
            # Ensure controller is properly released
            controller.release(keyboard.Key.space)

if __name__ == '__main__':
    unittest.main() 