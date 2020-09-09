var attack_steps = [
    "Slow (~50s / unique face) = Smaller Overall Distortion",
    "Quick (~10s / unique face) = Larger Overall Distortion",
];

var pert_steps = [
    "Heavy Distortion = Most Privacy",
    "More Distortion = More Privacy",
    "Some Distortion = Moderate Privacy",
    "Less Distortion = Less Privacy",
    "Light Distortion = Least Privacy",
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