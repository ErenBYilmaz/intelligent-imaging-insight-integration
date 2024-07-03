import os
from typing import List

from lib.my_logger import logging
from pydicom import dcmread
from pydicom.dataset import Dataset
from pynetdicom import AE, StoragePresentationContexts, VerificationPresentationContexts


class SenderConfiguration:
    def __init__(self, input_directory: str, pacs_address: str, pacs_port: int, pacs_aet: str):
        self.input_directory = input_directory
        self.pacs_address = pacs_address
        self.pacs_port = pacs_port
        self.pacs_aet = pacs_aet


class SendToLocalPython(SenderConfiguration):
    def __init__(self):
        super().__init__(input_directory="resources/example_dcm",
                         pacs_address='127.0.0.1',
                         pacs_port=104,
                         pacs_aet='PYTHON_AET', )


class SendToNicosOrthanC(SenderConfiguration):
    def __init__(self):
        super().__init__(input_directory="resources/example_dcm",
                         pacs_address='192.168.10.200',
                         pacs_port=104,
                         pacs_aet='ORTHANCA', )


class SendToNicosRaspberryPi(SenderConfiguration):
    def __init__(self):
        super().__init__(input_directory="resources/example_dcm",
                         pacs_address='192.168.10.203',
                         pacs_port=104,
                         pacs_aet='ORTHANCA', )


class Sender:
    def __init__(self, config: SenderConfiguration):
        self.config = config
        self.ae = AE('PYTHON_AET')
        self.ae.requested_contexts = StoragePresentationContexts[:-1] + VerificationPresentationContexts
        self.assoc = self.ae.associate(self.config.pacs_address, self.config.pacs_port, ae_title=self.config.pacs_aet)

    def send_dir(self, input_directory: str):

        # Create an association with the PACS server

        if self.assoc.is_established:
            logging.info(f"Association established with {self.config.pacs_address}:{self.config.pacs_port}")

            # Iterate over all DICOM files in the input directory
            for filename in os.listdir(input_directory):
                if filename.endswith(".dcm"):
                    filepath = os.path.join(input_directory, filename)
                    self.send_file(filepath)
        else:
            logging.error("Failed to establish association")

    def send(self, dcm_files: List[str]):
        if self.assoc.is_established:
            logging.info(f"Association established with {self.config.pacs_address}:{self.config.pacs_port}")

            # Iterate over all DICOM files in the input directory
            for file_path in dcm_files:
                self.send_file(file_path)
        else:
            logging.error("Failed to establish association")

    def __del__(self):
        self.assoc.release()

    def send_file(self, filepath):
        logging.info(f"Sending DICOM file: {filepath}")
        # Read the DICOM file
        ds = dcmread(filepath)
        # Send the DICOM file
        status = self.assoc.send_c_store(ds)
        # Check the status of the C-STORE operation
        if status:
            logging.info(f"C-STORE request status: 0x{status.Status:04x}")
        else:
            logging.error("Connection timed out, was aborted, or received invalid response")


def main():
    sender = Sender(SendToNicosOrthanC())
    # sender = Sender(SendToLocalPython())
    # sender.send_file(r"C:\Users\Eren\Programme\intelligent-imaging-insight-integration\temporary_files\received_dicom_files\a83db7f7-0b26-49c2-a92f-484c5c06bc98\1.2.276.0.7230010.3.1.2.2831156000.1.1499097860.742568\1.2.276.0.7230010.3.1.3.2831156000.1.1499097860.742569\DummySegmentationGenerator.dcm")
    # sender.send_dir(r"C:\Users\Eren\Programme\intelligent-imaging-insight-integration\resources\example_dcm")
    # sender.send_file(r"C:\Users\Eren\Programme\intelligent-imaging-insight-integration\temporary_files\received_dicom_files\11791306742903\1.2.276.0.50.192168001092.11156604.14547392.4\1.2.276.0.50.192168001092.11156604.14547392.195\DummySegmentationGenerator.dcm")
    # sender.send_dir(r"C:\Users\Eren\Downloads\04-01-2000-abdomenw-15076\2.000000-arterial-99348")
    sender.send_dir(r"C:\Users\Eren\Downloads\04-01-2000-abdomenw-15076\300.000000-Segmentation-99191")


if __name__ == '__main__':
    main()
