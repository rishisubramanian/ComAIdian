class AddFieldsToUsers < ActiveRecord::Migration[5.1]
  def change
    add_column :users, :gender, :string
    add_column :users, :age, :integer
    add_column :users, :birth_country, :string
    add_column :users, :major, :string
    add_column :users, :preferred_joke_genre, :string
    add_column :users, :preferred_joke_genre2, :string
    add_column :users, :preferred_joke_type, :string
    add_column :users, :favorite_music_genre, :string
    add_column :users, :favorite_movie_genre, :string
  end
end
