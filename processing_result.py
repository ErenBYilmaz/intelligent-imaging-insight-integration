import json
import os
import random
from typing import Dict, Any, List, Union

import SimpleITK
import pydicom
import pydicom_seg

from pdf2dcm_james import PDF2DCMProcessor


class ProcessingResult:
    def __init__(self, tool_name: str, metadata: Dict[str, Any] = None):
        self.tool_name = tool_name
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def to_dicom(self) -> List[str]:
        raise NotImplementedError('Abstract method')


class SegmentationResult(ProcessingResult):
    def __init__(self,
                 tool_name: str,
                 segmentation_nii_path: str,
                 metadata: Dict[str, Any] = None,
                 template_json_path: str = None,
                 base_dcm_dir_path: str = None, ):
        super().__init__(tool_name, metadata)
        self.segmentation_nii_path = segmentation_nii_path
        self.template_json_path = template_json_path
        self.base_dcm_dir_path = base_dcm_dir_path

    def dcm_paths(self, base_dcm_dir_path):
        return SimpleITK.ImageSeriesReader.GetGDCMSeriesFileNames(base_dcm_dir_path)

    def read_dcm_metadata(self, base_dcm_dir_path):
        dcm_files = self.dcm_paths(base_dcm_dir_path)
        metadata_reader = SimpleITK.ImageFileReader()
        metadata_reader.SetFileName(dcm_files[0])
        metadata_reader.LoadPrivateTagsOn()
        metadata_reader.ReadImageInformation()
        return metadata_reader

    def dcm_seg_template(self,
                         dcm_image_metadata: Union[SimpleITK.Image, SimpleITK.ImageFileReader]):
        with open(self.template_json_path) as f:
            template_dict = json.load(f)
        tag_map = {
            'ContentCreatorName': '0070|0084',
            'ClinicalTrialSeriesID': '0012|0071',
            'ClinicalTrialTimePointID': '0012|0050',
            'SeriesNumber': '0020|0011',
            'InstanceNumber': '0020|0013',
            'ClinicalTrialCoordinatingCenterName': '0012|0060',
            'BodyPartExamined': '0018|0015',
        }
        existing_keys = set(dcm_image_metadata.GetMetaDataKeys())
        for k in list(template_dict.keys()):
            if template_dict[k] is None:
                if tag_map[k] in existing_keys:
                    template_dict[k] = dcm_image_metadata.GetMetaData(tag_map[k])
                else:
                    del template_dict[k]
        template_dict['SegmentAlgorithmName'] = self.tool_name
        assert isinstance(template_dict['SeriesNumber'], str)
        template_dict['SeriesNumber'] = str(random.randint(template_dict['SeriesNumber'] + 1,
                                                           template_dict['SeriesNumber'] + 2 ** 20))
        template = pydicom_seg.template.from_dcmqi_metainfo(template_dict)
        return template

    def output_dcm_path(self):
        raise NotImplementedError('TO DO')

    def nii_to_dcm_seg(self):
        """
        Source https://razorx89.github.io/pydicom-seg/guides/write.html
        """
        to_dcm_seg_path = self.output_dcm_path()
        # dcm_image = self.read_dcm(base_dcm_dir_path)
        dcm_image_metadata = self.read_dcm_metadata(self.base_dcm_dir_path)

        writer = pydicom_seg.MultiClassWriter(
            template=self.dcm_seg_template(dcm_image_metadata),
            inplane_cropping=False,
            skip_empty_slices=False,
            skip_missing_segment=False,
        )

        segmentation_data = SimpleITK.ReadImage(self.segmentation_nii_path)

        dcm = writer.write(segmentation_data, self.dcm_paths(self.base_dcm_dir_path))
        dcm.save_as(to_dcm_seg_path)
        print('Wrote', os.path.abspath(to_dcm_seg_path))
        return to_dcm_seg_path

    def to_dicom(self) -> List[str]:
        # Example usage
        return [self.nii_to_dcm_seg()]


class PDFResult(ProcessingResult):
    def __init__(self,
                 tool_name: str,
                 pdf_path: str,
                 metadata: Dict[str, Any] = None):
        super().__init__(tool_name, metadata)
        self.pdf_path = pdf_path

    def to_dicom(self) -> List[str]:
        proc = PDF2DCMProcessor()
        return proc.fit(pdf_file_path=self.pdf_path, dcm_template_file_path=...)
