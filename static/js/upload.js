$(document).ready(function(){
    $("#upload-img-btn").click(function(){        
        $("#upload-img").submit(); // Submit the form
    });
});

$('#photo').live('change', (function(){
    var files = $(this)[0].files;
    if(files.length > 50){
        alert("you can select max 50 files.");
    }else{
        $('#file-counter').html(files.length + ' files selected');
    }
}));
