class ApplicationController < ActionController::Base
  protect_from_forgery with: :exception

  @full_width = false
end
