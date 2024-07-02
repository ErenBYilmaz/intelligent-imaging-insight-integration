from typing import Dict, Any, List


class ProcessingResult:
    def __init__(self, tool_name: str, metadata: Dict[str, Any] = None):
        self.tool_name = tool_name
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def to_dicom(self) -> List[str]:
        raise NotImplementedError('Abstract method')


class SegmentationResult(ProcessingResult):
    def to_dicom(self) -> List[str]:
        raise NotImplementedError('TO DO')


class PDFResult(ProcessingResult):
    def __init__(self, tool_name: str, pdf_path: str, metadata: Dict[str, Any] = None):
        super().__init__(tool_name, metadata)
        self.pdf_path = pdf_path

    def to_dicom(self) -> List[str]:
        raise NotImplementedError('TO DO')
