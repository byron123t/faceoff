var prev_window_width;
var prev_window_height;

$(document).live('ready', (function(){
    $("#upload-det-btn").click(function(){
        // $('.face-check:checked').each(function(){
        //     jqstring = 'hiddenSwitch' + $(this).attr('id');
        //     document.getElementById(jqstring).value = $(this).attr('id');
        // });
        $("#upload-det").submit(); // Submit the form
    });
}));

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

$(document).ready(function(){
    var window_width = $(window).width() - 40;
    var window_height = $(window).height() / 100 * 50;
    prev_window_width = window_width;
    prev_window_height = window_height;
    $('.det-image').each(function() {
        var width = $(this).attr('orig-width');
        var height = $(this).attr('orig-height');
        var aspect_ratio = width / height;
        var window_aspect_ratio = window_width / window_height;

        if(aspect_ratio > window_aspect_ratio) {
            var scale_width = window_width / width;
            var scale_height = window_width / aspect_ratio / height;
        } else {
            var scale_width = window_height * aspect_ratio / width;
            var scale_height = window_height / height;
        }
        $('[name ="bound' + $(this).attr('id') + '"]').each(function() {
            $(this).attr('orig-t', $(this).css('top'));
            $(this).attr('orig-l', $(this).css('left'));
            $(this).attr('orig-w', $(this).css('width'));
            $(this).attr('orig-h', $(this).css('height'));
        });
        if(scale_width < 1) {
            $('[name ="bound' + $(this).attr('id') + '"]').each(function() {
                $(this).css('top', $(this)[0].style.top.replace('px', '') * scale_height + 'px');
                $(this).css('left', $(this)[0].style.left.replace('px', '') * scale_width + 'px');
                $(this).css('width', $(this)[0].style.width.replace('px', '') * scale_width + 'px');
                $(this).css('height', $(this)[0].style.height.replace('px', '') * scale_height + 'px');
            });
        }
})});

$(window).on('resize', function(){
    var window_width = $(window).width() - 40;
    var window_height = $(window).height() / 100 * 50;
    prev_window_width = window_width;
    prev_window_height = window_height;
    $('.det-image').each(function() {
        var width = $(this).attr('orig-width');
        var height = $(this).attr('orig-height');
        var aspect_ratio = width / height;
        var window_aspect_ratio = window_width / window_height;

        if(aspect_ratio > window_aspect_ratio) {
            var scale_width = window_width / width;
            var scale_height = window_width / aspect_ratio / height;
        } else {
            var scale_width = window_height * aspect_ratio / width;
            var scale_height = window_height / height;
        }
        if(scale_width < 1) {
            $('[name ="bound' + $(this).attr('id') + '"]').each(function() {
                $(this).css('top', $(this).attr('orig-t').replace('px', '') * scale_height + 'px');
                $(this).css('left', $(this).attr('orig-l').replace('px', '') * scale_width + 'px');
                $(this).css('width', $(this).attr('orig-w').replace('px', '') * scale_width + 'px');
                $(this).css('height', $(this).attr('orig-h').replace('px', '') * scale_height + 'px');
            });
        } else {
            $('[name ="bound' + $(this).attr('id') + '"]').each(function() {
                $(this).css('top', $(this).attr('orig-t'));
                $(this).css('left', $(this).attr('orig-l'))
                $(this).css('width', $(this).attr('orig-w'));
                $(this).css('height', $(this).attr('orig-h'));
            });
        }
})});
