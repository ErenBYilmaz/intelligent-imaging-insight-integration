import json
import os
import unittest

import SimpleITK
import pydicom

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

    def test_read_metadata_with_pydicom(self):
        with open(r"C:\Users\Eren\Downloads\c31a32aa-ef01-4b42-80f3-74644305a6df.dcm", 'rb') as infile:
            ds = pydicom.dcmread(infile)

    def test_template(self):
        with open(os.path.join(examples_path, 'totalsegmentator_dcm_seg_template.json')) as f:
            template = json.load(f)
        assert len(template["segmentAttributes"]) == 74
        assert template["segmentAttributes"][-1]["SegmentedPropertyTypeCodeSequence"]["CodeValue"] == "68455001"