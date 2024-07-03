# (i²)² Framework

## How to connect an image processing tool to a PACS

(Instructions unfinished, use at own risk)

1. Clone this repository
2. Create a subclass of ImageProcessingResult or pick an existing one for storing the results
3. Create a subclass of ImageProcessingTool that implements the (AI-based or other) image processing and outputs the ImageProcessingResult from step 2
4. Create a Dockerfile containing the dependencies required for running the tool
    1. The following command installs packages in bulk according to the configuration file, requirements.txt. In some environments, use pip3 instead of pip.
    ```
    $ pip install -r requirements.txt
    ```
6. Build the corresponding Docker image
7. Start the container and inside the container start the dicom_receiver.py
