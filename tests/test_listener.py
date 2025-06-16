
import sys
sys.path.append('.')
import unittest
import queue
import threading
import time
import os
import tempfile
from demo.minimal_aoi import AOIListener, AOIEvent, AudioRecorder

class TestListener(unittest.TestCase):
    def setUp(self):
        # Create queues for the listener
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.listener = AOIListener(self.input_queue, self.output_queue)
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up
        self.listener.stop()
        # Remove temporary directory and its contents
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_recorder_initialization(self):
        """Test that the AudioRecorder initializes properly"""
        self.assertIsNotNone(self.listener.recorder)
        self.assertEqual(self.listener.recorder.CHANNELS, 1)
        self.assertEqual(self.listener.recorder.RATE, 44100)
        self.assertEqual(self.listener.recorder.FORMAT, self.listener.recorder.p.get_format_from_width(2))  # paInt16

    def test_space_key_control(self):
        """Test that space key press/release triggers appropriate events"""
        # Start the listener
        self.listener.start()
        time.sleep(0.1)  # Give it time to initialize
        
        # Simulate space key press
        self.listener._on_space_press(None)
        
        # Check for recording_start event
        try:
            event = self.output_queue.get(timeout=1.0)
            self.assertEqual(event.meta_data["type"], "recording_start")
            self.assertEqual(event.meta_data["sent_from"], "Listener")
        except queue.Empty:
            self.fail("No recording_start event received after space press")
        
        # Verify recording has started
        self.assertTrue(self.listener.recorder.recording)
        
        # Simulate space key release
        self.listener._on_space_release(None)
        
        # Check for recording_stop event
        try:
            event = self.output_queue.get(timeout=1.0)
            self.assertEqual(event.meta_data["type"], "recording_stop")
            self.assertEqual(event.meta_data["sent_from"], "Listener")
        except queue.Empty:
            self.fail("No recording_stop event received after space release")
        
        # Verify recording has stopped
        self.assertFalse(self.listener.recorder.recording)

    def test_audio_recording(self):
        """Test that audio recording actually captures data"""
        # Start recording
        self.listener.recorder.start_recording()
        time.sleep(1.0)  # Record for 1 second
        
        # Stop recording and get the file
        audio_file = self.listener.recorder.stop_recording()
        
        # Verify the file exists and has content
        self.assertTrue(os.path.exists(audio_file))
        self.assertGreater(os.path.getsize(audio_file), 0)
        
        # Clean up
        os.unlink(audio_file)

    def test_recording_cleanup(self):
        """Test that recording resources are properly cleaned up"""
        # Start and stop recording
        self.listener.recorder.start_recording()
        time.sleep(0.1)
        self.listener.recorder.stop_recording()
        
        # Verify stream is closed
        self.assertIsNone(self.listener.recorder.stream)
        
        # Call cleanup
        self.listener.recorder.cleanup()
        
        # Verify PyAudio is terminated
        self.assertFalse(self.listener.recorder.p._is_running)

if __name__ == '__main__':
    unittest.main() 