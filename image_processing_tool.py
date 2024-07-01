from image import Image
from processing_result import ProcessingResult


class ImageProcessingTool:
    def name(self) -> str:
        return type(self).__name__

    def can_process_image(self, img: Image) -> bool:
        raise NotImplementedError('Abstract method')

    def process(self, img: Image) -> ProcessingResult:
        raise NotImplementedError('Abstract method')