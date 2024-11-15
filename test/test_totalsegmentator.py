import os.path
import unittest

from examples.tools.totalsegmentator_tool import TotalSegmentator
from paths import resources_path


class TestTotalSegmentatorTool(unittest.TestCase):
    def test_total_segmentator(self):
        from image import Image

        tool = TotalSegmentator()
        img = Image('dummy.nii.gz',
                    metadata={'example_metadata_key': 'value'},
                    base_dcm_dir=os.path.join(resources_path, 'example_dcm'))
        assert tool.segmentation_result(img, '...').dcm_seg_template() is not None