$(document).live('ready', function(){
    $("#download-btn").click(function(){
    	$("#download-input").val("yes");
    	$("#doagain-input").val("no");
        $("#download").submit(); // Submit the form
    });
});

$(document).live('ready', function(){
    $("#doagain-btn").click(function(){
    	$("#download-input").val("no");
    	$("#doagain-input").val("yes");
        $("#doagain").submit(); // Submit the form
    });
});