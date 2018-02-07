# == Schema Information
#
# Table name: JokeRating
#
#  id            :integer          not null, primary key
#  rating        :integer          not null
#  joke_id       :integer
#  joke_rater_id :integer
#

# Stores administrative settings (dynamic config values accessible through admin tools)
class JokeRating < ApplicationRecord
  self.table_name = "JokeRating"
end
