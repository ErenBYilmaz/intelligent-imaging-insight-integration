import inspect
import json
import os
from typing import List, Union

import SimpleITK
import pydicom_seg

from image import Image
from processing_result import ProcessingResult


class ImageProcessingTool:
    def name(self) -> str:
        return type(self).__name__

    def version_id(self) -> str:
        raise NotImplementedError('Abstract method')

    def description(self) -> str:
        raise NotImplementedError('Abstract method')

    def can_process_image(self, img: Image) -> bool:
        raise NotImplementedError('Abstract method')

    def process(self, images: List[Image]) -> ProcessingResult:
        raise NotImplementedError('Abstract method')

    def mask_to_image(self, mask, base_image: SimpleITK.Image):
        mask_img = SimpleITK.GetImageFromArray(mask)
        mask_img.SetSpacing(base_image.GetSpacing())
        mask_img.SetOrigin(base_image.GetOrigin())
        mask_img.SetDirection(base_image.GetDirection())
        return mask_img
