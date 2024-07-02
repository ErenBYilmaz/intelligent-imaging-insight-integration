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
orthanc.ExtendOrthancExplorer('''
$( document ).ready(function() {
  setInterval(() => {
    if(!document.getElementById('sample-python-button') && document.getElementById('stow-study')){
        $('#sample-python-button').remove();                              
        var b = $('<a>')
        .attr('id', 'sample-python-button')
        .attr('data-role', 'button')
        .attr('href', '#')
        .attr('data-icon', 'forward')
        .attr('data-theme', 'a')
        .text('Execute AI Assist')
        .button()
        .click(function(e) {
            let copyButton = document.getElementById("study-archive-link")
            if(copyButton!==undefined){
                copyButton.click()
                setTimeout(() => {
                    navigator.clipboard.read().then((clipboardContents)=>{
                        for (const item of clipboardContents) {
                            if (!item.types.includes("text/plain")) {
                                throw new Error("Clipboard does not contain PNG image data.");
                            }
                            item.getType("text/plain").then(blob => {
                                console.log(blob)
                              blob.text().then(url => {
                                $.post('../execute-python', url, 
                              
                                    function(answer) {
                                        /*put your code after python here*/
                                        alert('Das hat alles wunderbar geklappt! YAY!');
                                    });
                              
                                
                              })
                            });
                        }      
                           
                            
                    
                     
                    });        
                },100)
                
            }
            
        });

        b.insertAfter($('#stow-study'));                      
    } 
                                                 
                            
  }, 200) ;                           
  
});                 

''')
