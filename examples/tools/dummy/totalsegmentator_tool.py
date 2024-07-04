import os
import subprocess
from importlib.metadata import version
from typing import List

import pydicom

from image import Image
from image_processing_tool import ImageProcessingTool
from totalsegmentator.python_api import totalsegmentator
from paths import examples_path
from processing_result import ProcessingResult, SegmentationResult


class TotalSegmentator(ImageProcessingTool):

    def version_id(self) -> str:
        return '1.' + version('totalsegmentator')

    def description(self) -> str:
        return 'Totalsegmentator (https://github.com/wasserth/TotalSegmentator)'

    def can_process_image(self, img: Image) -> bool:
        # Read the DICOM file
        example_dicom_file = img.dcm_slice_paths()[0]
        ds = pydicom.dcmread(example_dicom_file)

        # Check the Modality tag
        if ds.Modality == "CT":
            return True
        return False

    def process(self, images: List[Image]) -> ProcessingResult:
        assert len(images) == 1
        img = images[0]
        mask_img_path = 'total_segmentator_output.nii.gz'
        subprocess.check_output(['TotalSegmentator', '-i', img.nii_path, '-o', mask_img_path])
        # totalsegmentator(img.nii_path, mask_img_path)
        return SegmentationResult(tool_name=self.name(),
                                  segmentation_nii_path=mask_img_path,
                                  template_json_path=os.path.join(examples_path, 'totalsegmentator_dcm_seg_template.json'),
                                  base_dcm_dir_path=img.base_dcm_dir)
