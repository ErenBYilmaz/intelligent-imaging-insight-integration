from typing import Dict, Any, List


class ProcessingResult:
    def __init__(self, tool_name: str, metadata: Dict[str, Any]=None):
        self.tool_name = tool_name
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def to_dicom(self) -> List[str]:
        raise NotImplementedError('TO DO')