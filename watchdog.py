import os
import threading
import time
from math import inf
from typing import List, Set, Optional

from dicom_sender import SenderConfiguration
from examples.tools.dummy.dummy import DummyImageProcessingTool
from image import Image
from image_processing_tool import ImageProcessingTool
from lib.my_logger import logging
from lib.util import listdir_fullpath
from paths import temporary_files_path


class WatchDog(threading.Thread):
    def __init__(self,
                 image_processing_tool: ImageProcessingTool,
                 interval=1.,
                 base_received_images_path='temporary_files/received_dicom_files',
                 send_to: Optional[SenderConfiguration] = None,
                 daemon=False):
        threading.Thread.__init__(self, daemon=daemon)
        self.interval = interval
        self.stop = False
        self.image_processing_tool = image_processing_tool
        self.base_received_images_path = base_received_images_path
        self.known_received_files: Set[str] = set()
        self.latest_time_of_new_received_files = -inf
        self.send_to = send_to

    def run(self):
        while not self.stop:
            image_paths = self.unprocessed_image_paths()
            if len(image_paths) == 0:
                continue
            print(f'Nothing received since {self.nothing_received_since():.1f}s', )
            if self.nothing_received_for_60_seconds():
                for patient_id in self.received_patient_ids():
                    filtered_paths = [p for p in image_paths if patient_id in p]
                    logging.info(f'Processing images of patient {patient_id}.')
                    images = [Image.from_dcm_directory(image_path) for image_path in filtered_paths]
                    result = self.image_processing_tool.process(images)
                    logging.info(f'Done processing {len(images)} images of patient {patient_id}.')
                    dcm_paths = result.to_dicom()
                    if self.send_to is not None:
                        self.send_to.send(dcm_paths)
                    for image_path in filtered_paths:
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
                            logging.info(f'New file received: {file}')
                            self.known_received_files.add(file)
                            self.latest_time_of_new_received_files = time.time()
                    results.append(series_path)
        return results

    def nothing_received_for_60_seconds(self):
        return self.nothing_received_since() > 60

    def nothing_received_since(self):
        current_time = time.time()
        return current_time - self.latest_time_of_new_received_files

    def received_patient_ids(self):
        return os.listdir(self.base_received_images_path)

    def stop(self):
        self.stop = True


def main():
    tool = DummyImageProcessingTool()
    dog = WatchDog(tool, base_received_images_path=temporary_files_path, daemon=False)
    print('Starting WatchDog')
    dog.start()


if __name__ == '__main__':
    main()
