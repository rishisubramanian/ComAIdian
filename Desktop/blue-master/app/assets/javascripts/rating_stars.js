$(function () {

    var updateField = function(joke_id, rating) {
        $("input[type='hidden'][name=joke-rating-" + joke_id + "]").val(rating).trigger('change');
    };

    var ajax_submit = function(joke_id, rating) {
        $.ajax({
            type: "POST",
            url: '/joke_ratings',
            data: {
                rating: {
                    joke_id: joke_id,
                    rating: rating
                }
            }
        });
    };

    $('.rateYo').each(function() {
        $(this).rateYo({
            starWidth: "30px",
            fullStar: true,
            multiColor: true,
            onSet: function(rating, rateYoInstance) {
                updateField($(this).data('joke-id'),rating);

                if ($(this).hasClass('ajax_submit')) {
                    ajax_submit($(this).data('joke-id'), rating)
                }
            }
        });
    });


});