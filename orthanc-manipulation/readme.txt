Start orthanc with mount bindings like so

docker run -p 4242:4242 -p 8042:8042 --rm \
  -v /path/to/orthanc.json:/etc/orthanc/orthanc.json:ro \
  -v /path/to/aibutton.py:/etc/orthanc/aibutton.py:ro \
  jodogne/orthanc-python
