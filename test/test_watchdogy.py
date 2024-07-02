import math
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
