import os.path
from typing import Any, Dict, List

import SimpleITK

from lib.util import listdir_fullpath
from paths import temporary_files_path


class Image:
    def __init__(self, nii_path, base_dcm_dir: str, metadata: Dict[str, Any]):
        self.nii_path = nii_path
        self.metadata = metadata
        self.base_dcm_dir = base_dcm_dir

    def as_sitk_image(self) -> SimpleITK.Image:
        return SimpleITK.ReadImage(self.nii_path)

    def as_numpy(self):
        return SimpleITK.GetArrayFromImage(self.as_sitk_image())

    @classmethod
    def from_dcm_directory(cls, dcm_slices_dir: str, extra_metadata: Dict[str, Any] = None):
        nii_path = os.path.join(dcm_slices_dir, 'image.nii.gz')
        reader = SimpleITK.ImageSeriesReader()
        reader.SetFileNames(cls._dcm_slice_paths(dcm_slices_dir))
        s_image: SimpleITK.Image = reader.Execute()
        SimpleITK.WriteImage(s_image, nii_path)
        return Image(
            nii_path=nii_path,
            base_dcm_dir=dcm_slices_dir,
            metadata=extra_metadata if extra_metadata is not None else {},
        )

    @classmethod
    def _dcm_slice_paths(cls, dcm_slices_dir):
        return SimpleITK.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_slices_dir)

    def dcm_slice_paths(self):
        return self._dcm_slice_paths(self.base_dcm_dir)