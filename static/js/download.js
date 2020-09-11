$(document).live('ready', function(){
    check_progress($("#sess-id").html());
});

async function check_progress(sess_id) {
    console.log(sess_id);
    var source = new EventSource("/progress/"+sess_id);
    source.onmessage = function (event) {
        console.log(event.data);
        $("#prog").val(event.data);
        $("#prog").html = event.data+"%";
        if (event.data >= 100) {
            source.close();
            $("#done").submit();
        }
    }
}
