import orthanc
import platform

print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')
print('Hello world!')


def ExecutePython(output, uri, **request):
    s = 'Python version: %s' % platform.python_version()
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
                    

  
