import os.path
from typing import List

import SimpleITK
import numpy

from image import Image
from image_processing_tool import ImageProcessingTool
from lib.util import listdir_fullpath
from paths import examples_path
from processing_result import ProcessingResult, SegmentationResult


class DummyProcessingResult(ProcessingResult):
    def to_dicom(self):
        return listdir_fullpath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                             'resources', 'example_dcm'))


class DummyImageProcessingTool(ImageProcessingTool):
    def can_process_image(self, img: Image):
        return True

    def process(self, images: List[Image]) -> ProcessingResult:
        return DummyProcessingResult(self.name(), {'example_metadata_value': '42'})

    def description(self) -> str:
        return 'Dummy processing tool. Only returns example DICOM files.'

    def version_id(self) -> str:
        return '1.0.0'


class DummySegmentationGenerator(DummyImageProcessingTool):

    def process(self, images: List[Image]) -> ProcessingResult:
        assert len(images) == 1
        img = images[0]
        a = img.as_numpy()
        mask = numpy.random.randint(0, 2, a.shape, dtype=numpy.uint8)
        mask_img = self.mask_to_image(mask, img.as_sitk_image())
        mask_img_path = os.path.join(img.base_dcm_dir, 'dummy_seg.nii.gz')
        SimpleITK.WriteImage(mask_img, mask_img_path)
        return SegmentationResult(tool_name=self.name(),
                                  segmentation_nii_path=mask_img_path,
                                  template_json_path=os.path.join(examples_path, 'dcm_seg_template.json'),
                                  base_dcm_dir_path=img.base_dcm_dir)
