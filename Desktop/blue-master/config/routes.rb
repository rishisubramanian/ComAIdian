Rails.application.routes.draw do
  get 'joke_ratings/create'

  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html

  match '/visualization', to: 'visualization#index', via: 'get'

  match '/about_us', to: 'about_us#index', via: 'get'

  # Custom sign up form
  devise_for :users, :controllers => { registrations: 'registrations' }

  root 'welcome#index'

  match '/welcome', to: 'welcome#index', via: 'get'

  authenticate :user do
    resources :jokes, only: [:index]
    resources :joke_raters, only: [:new, :create]
    resources :joke_ratings, only: [:create]
  end
end
