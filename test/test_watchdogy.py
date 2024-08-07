import math
import time
import unittest

from examples.tools.dummy.dummy import DummyImageProcessingTool
from paths import temporary_files_path
from watchdog import WatchDog


class TestWatchDog(unittest.TestCase):
    def test_dummy(self):
        tool = DummyImageProcessingTool()
        dog = WatchDog(tool, base_received_images_path=temporary_files_path)
        assert len(dog.unprocessed_image_paths()) > 0
        assert dog.latest_time_of_new_received_files != -math.inf
        assert len(dog.received_patient_ids()) > 0

    def test_modification_time(self):
        tool = DummyImageProcessingTool()
        dog = WatchDog(tool, base_received_images_path=temporary_files_path)
        assert len(dog.unprocessed_image_paths()) > 0
        assert not dog.nothing_received_since() > 60
        dog.latest_time_of_new_received_files -= 61
        assert dog.nothing_received_since() > 60

    def test_watching(self):
        tool = DummyImageProcessingTool()
        dog = WatchDog(tool, base_received_images_path=temporary_files_path, daemon=True)
        print('Starting WatchDog')
        dog.start()
        time.sleep(2)
