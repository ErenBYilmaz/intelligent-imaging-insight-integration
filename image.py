from typing import Any, Dict, List

import SimpleITK


class Image:
    def __init__(self, nii_path, metadata: Dict[str, Any]):
        self.nii_path = nii_path
        self.metadata = metadata

    def as_sitk_image(self) -> SimpleITK.Image:
        return SimpleITK.ReadImage(self.nii_path)

    def from_dcm_slices(self, dcm_slice_paths: List[str], nii_path, extra_metadata: Dict[str, Any] = None):
        raise NotImplementedError('TO DO')
