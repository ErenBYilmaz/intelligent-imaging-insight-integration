import os
import subprocess
from importlib.metadata import version
from typing import List

import SimpleITK
import pydicom

from image import Image
from image_processing_tool import ImageProcessingTool
from paths import examples_path
from processing_result import ProcessingResult, SegmentationResult, MultipleResults


class TotalSegmentator(ImageProcessingTool):
    DISCARD_SEGMENTATIONS_ABOVE = 5

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
        results = []
        for img in images:
            mask_img_path = os.path.join(img.base_dcm_dir, 'total_segmentator_output.nii.gz')
            if not os.path.exists(mask_img_path):
                nii_dir = os.path.dirname(img.nii_path)
                nii_name = os.path.basename(img.nii_path)
                # Official command "docker run --gpus 'device=0' --ipc=host -v /absolute/path/to/my/data/directory:/tmp wasserth/totalsegmentator:2.2.1 TotalSegmentator -i /tmp/ct.nii.gz -o /tmp/segmentations"
                # subprocess.check_output(['docker', 'run', '--gpus', 'all', '--ipc=host', f'-v "{nii_dir}":/nii"', f'-v "{img.base_dcm_dir}":/dcm', f'wasserth/totalsegmentator:2.2.1', f'TotalSegmentator', f'-i /nii/{nii_name}', f'-o /dcm/total_segmentator_output.nii.gz', f'--rm', '--ml', '--fast'])
                subprocess.check_output([r'C:\ProgramData\Anaconda3\envs\Scripts\totalsegmentator', '-i', img.nii_path, '-o', mask_img_path, '--ml', '--fast'])
            processed_mask_path = os.path.join(img.base_dcm_dir, 'total_segmentator_output_processed.nii.gz')
            sitk_img = SimpleITK.ReadImage(mask_img_path)
            a = SimpleITK.GetArrayFromImage(sitk_img)
            a[a > 74] = 0
            a[a > self.DISCARD_SEGMENTATIONS_ABOVE] = 0
            processed_mask_img = SimpleITK.GetImageFromArray(a)
            processed_mask_img.CopyInformation(sitk_img)
            SimpleITK.WriteImage(processed_mask_img, processed_mask_path)
            results.append(self.segmentation_result(img, processed_mask_path))
        return MultipleResults(tool_name=self.name(), results=results)

    def segmentation_result(self, img, mask_img_path):
        return SegmentationResult(tool_name=self.name(),
                                  segmentation_nii_path=mask_img_path,
                                  template_json_path=os.path.join(examples_path, 'totalsegmentator_dcm_seg_template.json'),
                                  base_dcm_dir_path=img.base_dcm_dir,
                                  ignore_segmentations_above=self.DISCARD_SEGMENTATIONS_ABOVE)
