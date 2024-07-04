import os.path
import unittest

from examples.tools.dummy.dummy import DummyImageProcessingTool
from examples.tools.dummy.totalsegmentator_tool import TotalSegmentator
from paths import resources_path


class TestTotalSegmentatorTool(unittest.TestCase):
    def test_total_segmentator(self):
        from image import Image

        tool = TotalSegmentator()
        img = Image('dummy.nii.gz',
                    metadata={'example_metadata_key': 'value'},
                    base_dcm_dir=os.path.join(resources_path, 'example_dcm'))
        self.assertTrue(tool.can_process_image(img))
        result = tool.process([img])
        self.assertEqual(result.tool_name, 'DummyImageProcessingTool')
        self.assertEqual(result.metadata, {'example_metadata_value': '42'})
        dcm_output = result.to_dicom()
        assert len(dcm_output) > 0
        for slice in dcm_output:
            assert slice.endswith('.dcm')

    def test_version_present(self):
        tool = DummyImageProcessingTool()
        assert '0' in tool.version_id()

    def test_description_present(self):
        tool = DummyImageProcessingTool()
        assert len(tool.description()) > 0
