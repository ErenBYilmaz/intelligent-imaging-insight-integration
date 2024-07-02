import os.path
from typing import Any, Dict, List

import SimpleITK

from lib.util import listdir_fullpath
from paths import temporary_files_path


class Image:
    def __init__(self, nii_path, metadata: Dict[str, Any]):
        self.nii_path = nii_path
        self.metadata = metadata

    def as_sitk_image(self) -> SimpleITK.Image:
        return SimpleITK.ReadImage(self.nii_path)

    @classmethod
    def from_dcm_slices(cls, dcm_slice_paths: List[str], nii_path, extra_metadata: Dict[str, Any] = None):
        raise NotImplementedError('TO DO')

    @classmethod
    def from_dcm_directory(cls, dcm_slices_dir: str, extra_metadata: Dict[str, Any] = None):
        dcm_slices = [f for f in listdir_fullpath(dcm_slices_dir)
                      if os.path.isfile(f) and f.endswith('.dcm')]
        nii_path = os.path.join(dcm_slices_dir, 'image.nii')
        return cls.from_dcm_slices(dcm_slices, nii_path, extra_metadata)
