import os
import logging
from pydicom.dataset import Dataset
from pynetdicom import AE, evt, StoragePresentationContexts, VerificationPresentationContexts
from pynetdicom.sop_class import VerificationServiceClass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pynetdicom')
logger.setLevel(logging.DEBUG)

# Define the directory to save the incoming DICOM files
output_directory = "temporary_files/received_dicom_files"
os.makedirs(output_directory, exist_ok=True)


# Implement a handler for EVT_C_STORE
def handle_store(event):
    """Handle a C-STORE request event."""
    ds = event.dataset
    ds.file_meta = event.file_meta

    # Create a filename based on the SOP Instance UID
    filename = os.path.join(output_directory, f"{ds.SOPInstanceUID}.dcm")

    # Save the DICOM file
    ds.save_as(filename, write_like_original=False)
    logger.info(f"Stored DICOM file: {filename}")

    # Return a Success status
    return 0x0000


def main():
    # Set up the application entity (AE)
    ae = AE()

    # Add the supported presentation contexts
    ae.supported_contexts = StoragePresentationContexts + VerificationPresentationContexts

    # Define the handlers for the events
    handlers = [
        (evt.EVT_C_STORE, handle_store)
    ]

    # Start listening for incoming association requests
    print("Starting DICOM node on port 104...")
    ae.start_server(('', 104), evt_handlers=handlers, block=True)


if __name__ == '__main__':
    main()
