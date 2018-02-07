class RegistrationsController < Devise::RegistrationsController

  private

  # Override sign_up_params so we can have our own custom fields in it
  def sign_up_params
    params.require(:user).permit(:first_name, :last_name, :email, :password, :password_confirmation, :gender, :age,
                                 :birth_country, :major, :preferred_joke_genre, :preferred_joke_genre2,
                                 :preferred_joke_type, :favorite_music_genre, :favorite_movie_genre)
  end

  def account_update_params
    params.require(:user).permit(:first_name, :last_name, :email, :password, :password_confirmation, :current_password,
                                :gender, :age, :birth_country, :major, :preferred_joke_genre, :preferred_joke_genre2,
                                :preferred_joke_type, :favorite_music_genre, :favorite_movie_genre)
  end
end