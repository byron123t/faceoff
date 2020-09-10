$(document).live('ready', function(){
	$("#auto-input").val("yes");
	$("#done-input").val("no");
    $("#auto").submit();
    $("#auto-input").val("no");
    $("#done-input").val("yes");
    check_progress($("#sess-id").html());
});

async function check_progress(sess_id) {
    console.log(sess_id);
    var source = new EventSource("/progress/"+sess_id);
    source.onmessage = function (event) {
        var perc = parseInt(event.data);
        console.log(perc);
        $("#prog").val(perc);
        if (perc <= 50) {
            $("#estimated").html = "Sit tight, your download will appear in a bit...";
        } else if (perc >= 50) {
            $("#estimated").html = "Your download will appear soon...";
        } else if (perc >= 75) {
            $("#estimated").html = "Your download is almost done...";
        }
        if (perc >= 100) {
            source.close()
            $("#done").submit();
        }
    }
}
