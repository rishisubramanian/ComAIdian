# == Schema Information
#
# Table name: Joke
#
#  id                :integer          not null, primary key
#  category          :string(100)      not null
#  joke_type         :string(100)      not null
#  subject           :string(100)      not null
#  joke_text         :text             not null
#  joke_submitter_id :integer
#  joke_source       :string(100)      not null
#

# Stores administrative settings (dynamic config values accessible through admin tools)
class Joke < ApplicationRecord
  self.table_name = "Joke"

  def self.get_categories
    Joke.select(:category).map(&:category).uniq
  end

  def self.get_types
    Joke.select(:joke_type).map(&:joke_type).uniq
  end
end
