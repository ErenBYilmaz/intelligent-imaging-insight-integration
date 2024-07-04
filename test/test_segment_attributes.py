import json
import os
import unittest

import SimpleITK
import pydicom
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor

from paths import examples_path


class TestReadSegmentAttributes(unittest.TestCase):
    def test_read_metadata_with_sitk(self):
        reader = SimpleITK.ImageFileReader()
        reader.SetFileName(r"C:\Users\Eren\Downloads\c31a32aa-ef01-4b42-80f3-74644305a6df.dcm")
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        for k in reader.GetMetaDataKeys():
            v = reader.GetMetaData(k)
            print(f'({k}) = = "{v}"')

    def test_read_metadata_from_nii_with_sitk(self):
        reader = SimpleITK.ImageFileReader()
        reader.SetFileName(r"C:\Users\Eren\Programme\intelligent-imaging-insight-integration\temporary_files\received_dicom_files\100002\1.2.840.113654.2.55.187766322555605983451267194286230980878\1.2.840.113654.2.55.97114726565566537928831413367474015470\total_segmentator_output.nii.gz")
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        for k in reader.GetMetaDataKeys():
            v = reader.GetMetaData(k)
            print(f'({k}) = = "{v}"')

    def test_read_processed_metadata_from_nii_with_sitk(self):
        reader = SimpleITK.ImageFileReader()
        reader.SetFileName(r"C:\Users\Eren\Programme\intelligent-imaging-insight-integration\temporary_files\received_dicom_files\100002\1.2.840.113654.2.55.187766322555605983451267194286230980878\1.2.840.113654.2.55.97114726565566537928831413367474015470\total_segmentator_output.nii.gz")
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        for k in reader.GetMetaDataKeys():
            v = reader.GetMetaData(k)
            print(f'({k}) = = "{v}"')

    def test_read_metadata_with_pydicom(self):
        with open(r"C:\Users\Eren\Downloads\c31a32aa-ef01-4b42-80f3-74644305a6df.dcm", 'rb') as infile:
            ds = pydicom.dcmread(infile)
        print(ds)

    def test_read_created_metadata_with_pydicom(self):
        with open(r"C:\Users\Eren\Programme\intelligent-imaging-insight-integration\temporary_files\received_dicom_files\100002\1.2.840.113654.2.55.187766322555605983451267194286230980878\1.2.840.113654.2.55.97114726565566537928831413367474015470\TotalSegmentator.dcm", 'rb') as infile:
            ds = pydicom.dcmread(infile)
        print(ds)

    def test_template(self):
        with open(os.path.join(examples_path, 'totalsegmentator_dcm_seg_template.json')) as f:
            template = json.load(f)
        assert len(template["segmentAttributes"]) == 74
        assert template["segmentAttributes"][-1]["SegmentedPropertyTypeCodeSequence"]["CodeValue"] == "68455001"

    def scale_lab_values(self, lab_values):
        scaled_values = [
            (lab_values[0] / 65535) * 100,  # L* value is typically between 0 and 100
            (lab_values[1] / 65535) * 255 - 128,  # a* value is typically between -128 and 127
            (lab_values[2] / 65535) * 255 - 128  # b* value is typically between -128 and 127
        ]
        return scaled_values

    def lab_to_rgb(self, lab_values):
        scaled_lab_values = self.scale_lab_values(lab_values)
        lab_color = LabColor(lab_l=scaled_lab_values[0], lab_a=scaled_lab_values[1], lab_b=scaled_lab_values[2])
        rgb_color = convert_color(lab_color, sRGBColor)
        rgb_values = [int(max(0, min(255, x * 255))) for x in (rgb_color.clamped_rgb_r, rgb_color.clamped_rgb_g, rgb_color.clamped_rgb_b)]
        return rgb_values
