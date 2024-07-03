function getQueryVariable(variable) {
    var query = window.location.href.split("?")[1]
    var vars = query.split("&");
    for (var i=0;i<vars.length;i++) {
      var pair = vars[i].split("=");
      if (pair[0] == variable) {
        return pair[1];
      }
    } 
    return undefined
  }


function executeAiAssist(study){
    if(study==="" || study===undefined){
        alert("study id konnte nicht geladen werden.")
    }
    else{

        $.post('../execute-python', study,                                       
            function(answer) {
                /*put your after python code here*/
                window.open("/ohif/viewer?url=../studies/"+getQueryVariable("uuid")+"/ohif-dicom-json")
                window.open("C:\Users\info\PycharmProjects\DentalMaskRcnn\results\example_tooth.png");              
                    
                
                alert('Das hat alles wunderbar geklappt! YAY!');
            }
        );
    }
    
}

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
            const uuid = getQueryVariable("uuid")
            executeAiAssist(uuid)
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
