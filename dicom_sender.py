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


def main(config: SenderConfiguration):
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('pynetdicom')
    logger.setLevel(logging.DEBUG)

    # Unpack the configuration
    input_directory = config.input_directory
    pacs_address = config.pacs_address
    pacs_port = config.pacs_port
    pacs_aet = config.pacs_aet

    # Set up the application entity (AE)
    ae = AE()

    # Add the requested presentation contexts
    ae.requested_contexts = StoragePresentationContexts[:-1] + VerificationPresentationContexts

    # Create an association with the PACS server
    assoc = ae.associate(pacs_address, pacs_port, ae_title=pacs_aet)

    if assoc.is_established:
        logger.info(f"Association established with {pacs_address}:{pacs_port}")

        # Iterate over all DICOM files in the input directory
        for filename in os.listdir(input_directory):
            if filename.endswith(".dcm"):
                filepath = os.path.join(input_directory, filename)
                logger.info(f"Sending DICOM file: {filepath}")

                # Read the DICOM file
                ds = dcmread(filepath)

                # Send the DICOM file
                status = assoc.send_c_store(ds)

                # Check the status of the C-STORE operation
                if status:
                    logger.info(f"C-STORE request status: 0x{status.Status:04x}")
                else:
                    logger.error("Connection timed out, was aborted, or received invalid response")

        # Release the association
        assoc.release()
        logger.info("Association released")
    else:
        logger.error("Failed to establish association")  #


if __name__ == '__main__':
    main(SendToNicosRaspberryPi())
