# == Schema Information
#
# Table name: JokeRater
#
#  id                    :integer          not null, primary key
#  joke_submitter_id     :string(100)      not null
#  gender                :string(100)      not null
#  birth_country         :string(100)      not null
#  major                 :string(100)      not null
#  preferred_joke_genre  :string(100)      not null
#  preferred_joke_genre2 :string(100)      not null
#  preferred_joke_type   :string(100)      not null
#  favorite_music_genre  :string(100)      not null
#  favorite_movie_genre  :string(100)      not null
#  age                   :integer
#

# Stores administrative settings (dynamic config values accessible through admin tools)
class JokeRater < ApplicationRecord
  self.table_name = "JokeRater"

  def self.get_music_genres
    JokeRater.select(:favorite_music_genre).map(&:favorite_music_genre).uniq
  end

  def self.get_movie_genres
    JokeRater.select(:favorite_movie_genre).map(&:favorite_movie_genre).uniq
  end

  def self.get_birth_countries
    JokeRater.select(:birth_country).map(&:birth_country).uniq.reject { |c| c.empty? }
  end

  def self.get_majors
    JokeRater.select(:major).map(&:major).uniq
  end
end
