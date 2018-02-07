# == Schema Information
#
# Table name: users
#
#  id                      :integer          not null, primary key
#  email                   :string           default(""), not null
#  encrypted_password      :string           default(""), not null
#  reset_password_token    :string
#  reset_password_sent_at  :datetime
#  remember_created_at     :datetime
#  sign_in_count           :integer          default(0), not null
#  current_sign_in_at      :datetime
#  last_sign_in_at         :datetime
#  current_sign_in_ip      :string
#  last_sign_in_ip         :string
#  created_at              :datetime         not null
#  updated_at              :datetime         not null
#  gender                  :string
#  age                     :integer
#  birth_country           :string
#  major                   :string
#  preferred_joke_genre    :string
#  preferred_joke_genre2   :string
#  preferred_joke_type     :string
#  favorite_music_genre    :string
#  favorite_movie_genre    :string
#  initial_rating_complete :boolean          default(FALSE)
#  joke_rater_id           :integer
#

require 'test_helper'

class UserTest < ActiveSupport::TestCase
  # test "the truth" do
  #   assert true
  # end
end
