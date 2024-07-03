import os
import threading
import time
from math import inf
from typing import List, Set

from image import Image
from image_processing_tool import ImageProcessingTool
from lib.util import listdir_fullpath
from paths import temporary_files_path


class WatchDog(threading.Thread):
    def __init__(self,
                 image_processing_tool: ImageProcessingTool,
                 interval=1.,
                 base_received_images_path='temporary_files/received_dicom_files',
                 daemon=False):
        threading.Thread.__init__(self, daemon=daemon)
        self.interval = interval
        self.stop = False
        self.image_processing_tool = image_processing_tool
        self.base_received_images_path = base_received_images_path
        self.known_received_files: Set[str] = set()
        self.latest_time_of_new_received_files = -inf

    def run(self):
        while not self.stop:
            image_paths = self.unprocessed_image_paths()
            if self.nothing_received_for_60_seconds():
                for _patient_id in self.received_patient_ids():
                    images = [Image.from_dcm_directory(image_path) for image_path in image_paths]
                    self.image_processing_tool.process(images)
            for image_path in image_paths:
                open(os.path.join(image_path, 'is_processed.txt'), 'a').close()
            time.sleep(self.interval)

    def unprocessed_image_paths(self):
        results = []
        for patient_id in self.received_patient_ids():
            patient_path = os.path.join(temporary_files_path, patient_id)
            for study_id in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_id)
                for series_id in os.listdir(study_path):
                    series_path = os.path.join(study_path, series_id)
                    if os.path.isfile(os.path.join(series_path, 'is_processed.txt')):
                        continue
                    for file in listdir_fullpath(series_path):
                        if file not in self.known_received_files:
                            self.known_received_files.add(file)
                            self.latest_time_of_new_received_files = time.time()
                    results.append(series_path)
        return results

    def nothing_received_for_60_seconds(self):
        current_time = time.time()
        return self.latest_time_of_new_received_files < current_time - 60

    def received_patient_ids(self):
        return os.listdir(self.base_received_images_path)

    def stop(self):
        self.stop = True
