$( document ).ready(function() {
  setInterval(() => {
    if(!document.getElementById('executeAiAssist') && document.getElementById('stow-study')){
        $('#executeAiAssist').remove();                              
        var b = $('<a>')
        .attr('id', 'executeAiAssist')
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
                                        /*put your after python code here*/
                                        window.open("C:\Users\info\PycharmProjects\DentalMaskRcnn\results\example_tooth.png");
                                        alert('Das hat alles wunderbar geklappt! YAY!');
                                    });
                              
                                
                              })
                            });
                        }      
                           
                            
                    
                     
                    });        
                },100)
                
            }
            
        });
        $(document).keydown(function(event) {
            if (event.altKey && event.which === 88)
            {
                b.click()
                e.preventDefault();
            }
        });
        b.insertAfter($('#stow-study'));                      
    } 
                                                 
                            
  }, 200) ;                           
  
});          
