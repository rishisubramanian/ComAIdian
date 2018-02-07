class JokeRatersController < ApplicationController
  include JokesHelper
  include JokeRatersHelper

  before_action :block_already_rated, only: [:new, :create]

  NUM_INITIAL_RATINGS = 5

  def new
    @page_title = 'Please rate these jokes before you proceed'
    @jokes = random_sample_jokes(NUM_INITIAL_RATINGS)
  end

  def create
    params = create_params
    params = parse_create_params params
    if validate_create_params params
      # Create joke rater for user. This should only ever happen once
      joke_rater = JokeRater.create!(
          joke_submitter_id: current_user.id,
          age: current_user.age,
          gender: current_user.gender,
          birth_country: current_user.birth_country,
          major: current_user.major,
          preferred_joke_genre: current_user.preferred_joke_genre,
          preferred_joke_genre2: current_user.preferred_joke_genre2,
          preferred_joke_type: current_user.preferred_joke_type,
          favorite_music_genre: current_user.favorite_music_genre,
          favorite_movie_genre: current_user.favorite_movie_genre
      )
      # Create new Joke ratings for each joke rated
      params.each{ |joke_id, rating|
        JokeRating.create!(
          rating: rating.to_i,
          joke_id: joke_id.to_i,
          joke_rater_id: joke_rater.id
        )
      }
      # User has finished initial rating process
      current_user.joke_rater = joke_rater
      current_user.update! initial_rating_complete: true
      redirect_to controller: 'jokes', action: 'index'
    else
      redirect_to action: 'new', notice: 'Invalid ratings detected'
    end

  end

  private

  # Block users from accessing the page who have already finished the initial rating process
  def block_already_rated
    if current_user.initial_rating_complete
      redirect_to controller: 'jokes', action: 'index'
    end
  end

  # Permit params that follow the pattern "joke-rating-ddd".
  def create_params
    permitted_params = params.keys.grep(/^joke-rating-\d+$/).map {|k| k.to_sym }
    params.permit(permitted_params)
  end

  # Turn params into hash of joke_id keys and rating values
  def parse_create_params(params)
    Hash[params.to_h.map {|k, v| [k[/\d+/], v]}]
  end

  # Make sure we recieve the correct number of params, and that they match joke ids in the database
  def validate_create_params(params)
    params.count == NUM_INITIAL_RATINGS && params.all? {|key, rating| Joke.exists?(key) && (1..5).include?(rating.to_i)}
  end
end
