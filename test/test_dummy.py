import unittest


class TestDummyProcessingTool(unittest.TestCase):
    def test_dummy(self):
        from tools.dummy import DummyImageProcessingTool, DummyProcessingResult
        from image import Image

        tool = DummyImageProcessingTool()
        img = Image('dummy.nii', {'example_metadata_key': 'value'})
        self.assertTrue(tool.can_process_image(img))
        result = tool.process(img)
        self.assertEqual(result.tool_name, 'DummyImageProcessingTool')
        self.assertEqual(result.metadata, {'example_metadata_value': '42'})
        dcm_output = result.to_dicom()
        assert len(dcm_output) > 0
        for slice in dcm_output:
            assert slice.endswith('.dcm')

    def test_version_present(self):
        from tools.dummy import DummyImageProcessingTool
        tool = DummyImageProcessingTool()
        assert '0' in tool.version_id()

    def test_description_present(self):
        from tools.dummy import DummyImageProcessingTool
        tool = DummyImageProcessingTool()
        assert len(tool.description()) > 0
