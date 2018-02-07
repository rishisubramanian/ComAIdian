// Make sure that users rate all the jokes before continuing
$( document ).on('turbolinks:load', function() {
    var rating_inputs = $("input[type='hidden'][name^='joke-rating-']");
    var submit_button = $('#submit-ratings');

    var check_all_rated = function() {
        var all_rated = true;
        rating_inputs.each(function() {
            if ($(this).val().length <= 0) {
                all_rated = false;
                return false;
            }
        });

        return all_rated;
    };

    submit_button.prop('disabled',true);

    rating_inputs.on('change', function() {
        if (check_all_rated()) {
            submit_button.prop('disabled',false);
        } else {
            submit_button.prop('disabled',true);
        }
    });
});
