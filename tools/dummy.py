import os.path

from image import Image
from image_processing_tool import ImageProcessingTool
from lib.util import listdir_fullpath
from processing_result import ProcessingResult


class DummyProcessingResult(ProcessingResult):
    def to_dicom(self):
        return listdir_fullpath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                             'resources', 'example_dcm'))


class DummyImageProcessingTool(ImageProcessingTool):
    def can_process_image(self, img: Image):
        return True

    def process(self, img: Image):
        return DummyProcessingResult(self.name(), {'example_metadata_value': '42'})

    def description(self) -> str:
        return 'Dummy processing tool. Only returns example DICOM files.'

    def version_id(self) -> str:
        return '1.0.0'
