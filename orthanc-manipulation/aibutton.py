import os
import urllib.request
import zipfile

import orthanc
import platform

from dicom_sender import Sender, SendToErensPython

print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')


def dicom_zip_url(base_url):
    """
    convert from format http://192.168.10.200:8042/app/explorer.html#study?uuid=f313764a-ff63c22b-cfa8ed77-45823e1f-c089412a
    to format http://192.168.10.200:8042/studies/f313764a-ff63c22b-cfa8ed77-45823e1f-c089412a/archive
    """
    return base_url.replace('app/explorer.html#study?uuid=', 'studies/') + '/archive'


def ExecutePython(output, uri, **request):
    zip_address = dicom_zip_url(uri)
    s = f'Python version: {platform.python_version()}, uri: {uri}, download_url: {zip_address}'
    urllib.request.urlretrieve(zip_address, "tmp.zip")
    if os.path.isdir("tmp"):
        import shutil
        shutil.rmtree("tmp")
    os.makedirs("tmp")
    with zipfile.ZipFile("tmp.zip", 'r') as zip_ref:
        zip_ref.extractall("tmp")
    sender = Sender(SendToErensPython())
    sender.send_dir_and_subdirs("tmp")
    print(request.__getitem__("body"))
    output.AnswerBuffer(s, 'text/plain')


orthanc.RegisterRestCallback('/execute-python', ExecutePython)
##adds a jQuery-Trigger on document ready that checks for the existence of the sample button every given time intervall 
# and whether the button to attach is present.
# then adds the AI Assist-Button if criteria met
with open('/etc/orthanc/aibutton.js', 'r') as file:
    data = file.read()
    print(data)
    orthanc.ExtendOrthancExplorer(data)
