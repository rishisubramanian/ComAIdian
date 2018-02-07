class AddJokeRaterIdFieldToUser < ActiveRecord::Migration[5.1]
  def change
    add_column :users, :joke_rater_id, :integer
  end
end
