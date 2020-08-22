$(document).ready(function(){
    $("#upload-img-btn").click(function(){        
        $("#upload-img").submit(); // Submit the form
    });
});

$(document).ready(function(){
    $("#upload-det-btn").click(function(){
        // $('.face-check:checked').each(function(){
        //     jqstring = 'hiddenSwitch' + $(this).attr('id');
        //     document.getElementById(jqstring).value = $(this).attr('id');
        // });
        $("#upload-det").submit(); // Submit the form
    });
});

$(document).ready(function(){
    $("#upload-pert-btn").click(function(){        
        $("#upload-pert").submit(); // Submit the form
    });
});

$(function(){
    $('#button').click(function(){
        var user = $('#inputUsername').val();
        var pass = $('#inputPassword').val();
        $.ajax({
            url: '/signUpUser',
            data: $('#form').serialize(),
            type: 'POST',
            success: function(response){
                console.log(response);
            },
            error: function(error){
                console.log(error);
            }
        });
    });
});

$('#select-all-btn').live('change', (function() {
    if($(this).is(':checked')) {
        // Iterate each checkbox
        $('.face-check').each(function() {
            this.checked = true;
        });
    } else {
        $('.face-check').each(function() {
            this.checked = false;
        });
    }
}));

function unselect(source) {
    if (!source.checked) {
        document.getElementById('select-all-btn').checked = false;
    }
}

$('.face-check').live('change', (function(){
    if ($('.face-check:checked').length == $('.face-check').length) {
       $('#select-all-btn').attr('checked', true);
    }
}));
