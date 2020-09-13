var attack_steps = [
    "Subtle Distortion<br/>(~30s / unique face)",
    "Visible Distortion<br/>(~10s / unique face)",
];

var pert_steps = [
    "Heavy Distortion<br/>Most Privacy",
    "More Distortion<br/>More Privacy",
    "Some Distortion<br/>Moderate Privacy",
    "Less Distortion<br/>Less Privacy",
    "Light Distortion<br/>Least Privacy",
];

$(document).ready(function(){
    $("#upload-pert-btn").click(function(){        
        $("#upload-pert").submit(); // Submit the form
    });
});

function attackUpdate(value) {
    $('#attack-display').html(attack_steps[value]);
}

function pertUpdate(value) {
    $('#pert-display').html(pert_steps[value]);
}

$(document).live('ready', (function(){
    $('#attack-slide')[0].classList.add('');
    $('#pert-slide')[0].classList.add('');
}));