class JokeRatingsController < ApplicationController
  def create
    p = create_params.to_h
    JokeRating.create(
        rating: p[:rating].to_i,
        joke_id: p[:joke_id].to_i,
        joke_rater_id: current_user.joke_rater.id
    )
  end

  private

  def create_params
    params.require(:rating).permit [:joke_id, :rating]
  end
end
