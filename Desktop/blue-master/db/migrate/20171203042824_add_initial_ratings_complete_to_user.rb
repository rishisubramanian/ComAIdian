class AddInitialRatingsCompleteToUser < ActiveRecord::Migration[5.1]
  def change
    add_column :users, :initial_rating_complete, :boolean, default: false
  end
end
