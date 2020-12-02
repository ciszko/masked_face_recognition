$(document).ready(function () {
    var check1 = document.getElementById("check1");

    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });


    $("#check1").change(function () {
        if (this.checked) {
            console.log("hejka")
        }
        $("#hidden_button").toggle();
    });

    $("#send-but").click(function () {
        $('#submit-but').click();
    });

    function previewImages() {

        var $preview = $('#preview').empty();
        if (this.files) $.each(this.files, readAndPreview);

        function readAndPreview(i, file) {

            if (!/\.(jpe?g|png|bmp)$/i.test(file.name)) {
                return alert(file.name + " is not an image");
            } // else...

            var reader = new FileReader();

            $(reader).on("load", function () {
                $preview.append($('<img/>', { src: this.result, height: 100 }));
            });

            reader.readAsDataURL(file);

        }

    }

    $('#file').on("change", previewImages);

});