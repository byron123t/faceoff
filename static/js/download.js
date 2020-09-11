$(document).live('ready', function(){
    check_progress($("#sess-id").html());
});

async function check_progress(sess_id) {
    if (sess_id != undefined) {
        console.log(sess_id);
        while(true) {
            $.get("/progress/"+sess_id, function(data) {
                console.log(data);
                $("#prog").val(data);
                $("#prog").html = data+"%";
                if (data >= 100) {
                    $("#done").submit();
                }
            });
        await new Promise(r => setTimeout(r, 2000));
        }
    }
}
