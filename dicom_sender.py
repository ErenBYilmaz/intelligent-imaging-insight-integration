import os
import logging
from pydicom import dcmread
from pydicom.dataset import Dataset
from pynetdicom import AE, StoragePresentationContexts, VerificationPresentationContexts


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('pynetdicom')
    logger.setLevel(logging.DEBUG)

    # Define the directory containing the DICOM files to send
    input_directory = "resources/example_dcm"

    # Define the destination PACS server details
    pacs_address = '127.0.0.1'
    pacs_port = 104
    pacs_aet = 'PYTHON_AET'

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
    main()
