import os
import threading
import time
from math import inf
from typing import Set, Optional

from dicom_sender import SenderConfiguration, Sender, SendToNicosOrthanC
from examples.tools.totalsegmentator_tool import TotalSegmentator
from image import Image
from image_processing_tool import ImageProcessingTool
from lib.my_logger import logging
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
        self.known_directories: Set[str] = set()
        self.latest_time_of_new_received_files = -inf
        self.send_to = send_to

    def run(self):
        while not self.stop:
            image_paths = self.unprocessed_image_paths()
            if len(image_paths) == 0:
                continue
            print(f'Nothing received since {self.nothing_received_since():.1f}s', )
            if self.nothing_received_since() > 10:
                for patient_id in self.received_patient_ids():
                    filtered_paths = [p for p in image_paths if patient_id in p]
                    if len(filtered_paths) == 0:
                        continue
                    logging.info(f'Processing images of patient {patient_id}.')
                    images = [Image.from_dcm_directory(image_path) for image_path in filtered_paths]
                    images = [img for img in images if self.image_processing_tool.can_process_image(img)]
                    if len(images) != 0:
                        result = self.image_processing_tool.process(images)
                        logging.info(f'Done processing {len(images)} images of patient {patient_id}.')
                        dcm_paths = result.to_dicom()
                        logging.info(f'Done converting to dicom. Created {dcm_paths}')
                        if self.send_to is not None:
                            Sender(self.send_to).send(dcm_paths)
                    for image_path in filtered_paths:
                        open(self.processing_note_path(image_path), 'a').close()
            time.sleep(self.interval)

    def processing_note_path(self, image_path):
        return os.path.join(image_path, f'was_processed_by_{self.tool_name()}.txt')

    def tool_name(self):
        return self.image_processing_tool.name()

    def unprocessed_image_paths(self):
        results = []
        for patient_id in self.received_patient_ids():
            patient_path = os.path.join(temporary_files_path, patient_id)
            for study_id in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_id)
                for series_id in os.listdir(study_path):
                    series_path = os.path.join(study_path, series_id)
                    if series_path in self.known_directories:
                        continue
                    if os.path.isfile(self.processing_note_path(series_path)):
                        self.known_directories.add(series_path)
                        continue
                    slices = Image._dcm_slice_paths(series_path)
                    if len(slices) == 0:
                        self.known_directories.add(series_path)
                        continue
                    for file in slices:
                        if file not in self.known_received_files:
                            logging.info(f'New file received: {file}')
                            self.known_received_files.add(file)
                            self.latest_time_of_new_received_files = time.time()
                    results.append(series_path)
        return results

    def nothing_received_since(self):
        current_time = time.time()
        return current_time - self.latest_time_of_new_received_files

    def received_patient_ids(self):
        return os.listdir(self.base_received_images_path)

    def stop(self):
        self.stop = True


def main():
    tool = TotalSegmentator()
    # sender = None
    config = SendToNicosOrthanC()
    # sender = Sender(config)
    dog = WatchDog(tool, base_received_images_path=temporary_files_path, daemon=False, send_to=config)
    print('Starting WatchDog')
    dog.start()


if __name__ == '__main__':
    main()
