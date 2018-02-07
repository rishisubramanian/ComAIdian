class JokesController < ApplicationController
  include JokesHelper
  before_action :require_initial_ratings, only: [:index]

  NUM_JOKES_TO_SHOW = 3

  def index
    @page_title = 'Your Recommended Jokes'
    begin
      joke_ids = predict_jokes(NUM_JOKES_TO_SHOW)
    rescue
      # If machine learning fails to return correctly, get random jokes
      joke_ids = random_sample_jokes(NUM_JOKES_TO_SHOW)
    end

    @jokes = Joke.where(id: joke_ids)
  end

  private

  def require_initial_ratings
    unless current_user.initial_rating_complete
      redirect_to controller: 'joke_raters', action: 'new'
    end
  end
end
