import logging
import os

from pynetdicom import AE, evt, StoragePresentationContexts, VerificationPresentationContexts


def main():
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

        # Extract patient ID, study instance UID, and series instance UID
        patient_id = ds.PatientID
        study_instance_uid = ds.StudyInstanceUID
        series_instance_uid = ds.SeriesInstanceUID

        # Create directory structure
        patient_directory = os.path.join(output_directory, patient_id)
        study_directory = os.path.join(patient_directory, study_instance_uid)
        series_directory = os.path.join(study_directory, series_instance_uid)

        # Ensure the directories exist
        os.makedirs(series_directory, exist_ok=True)

        # Create a filename based on the SOP Instance UID
        filename = os.path.join(series_directory, f"{ds.SOPInstanceUID}.dcm")

        # Save the DICOM file
        ds.save_as(filename, write_like_original=False)
        logger.info(f"Stored DICOM file: {filename}")

        # Return a Success status
        return 0x0000

    # Set up the application entity (AE)
    title = 'PYTHON_AET'
    ae = AE(ae_title=title)

    # Add the supported presentation contexts
    ae.supported_contexts = StoragePresentationContexts + VerificationPresentationContexts

    # Define the handlers for the events
    handlers = [
        (evt.EVT_C_STORE, handle_store)
    ]

    # Start listening for incoming association requests
    port = 104
    print(f"Starting DICOM node {title} on port {port}...")
    ae.start_server(('', port), evt_handlers=handlers, block=True)


if __name__ == '__main__':
    main()
