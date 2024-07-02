from pdf2dcm import Pdf2EncapsDCM


class PDF2DCMProcessor():

    def __init__(self):
        self.converter = Pdf2EncapsDCM()

    def fit(self, pdf_file_path, dcm_template_file_path):
        """
        path_pdf (str): path of the pdf that needs to be converted
        path_template_dcm (str, optional): path to template for getting the repersonalisation of data.
        suffix (str, optional): suffix of the dicom files. Defaults to ".dcm".
        """

        return self.converter.run(
            path_pdf=pdf_file_path,
            path_template_dcm=dcm_template_file_path,
            suffix=".dcm"
        )
