require 'net/http'

module JokesHelper
  # Pick a random sample of num jokes
  def random_sample_jokes(num)
    Joke.order("random()").first(num)
  end

  # Run machine learning predictor to find predicted jokes for user
  def predict_jokes(num)
    db_name = ActiveRecord::Base.connection.current_database

    if Rails.env.development? || Rails.env.test?
      db_host = ''
      db_username = ENV['USER']
      db_password = ''
      local = 'true'

      predictor_file = "#{Rails.root}/ML/matrix_completion/run_predictor.py"
      arguments = "\"#{num.to_s}\" \"#{current_user.joke_rater.id.to_s}\" \"#{db_name}\" \"#{db_host}\" \"#{db_username}\" \"#{db_password}\" \"#{local}\""
      # Call python script
      result = %x(python #{predictor_file} #{arguments})
    else
      config = ActiveRecord::Base.connection_config
      db_host = config[:host]
      db_username = config[:username]
      db_password = config[:password]
      local = 'false'

      rater_id = current_user.joke_rater.id.to_s
      url_base = 'http://ec2-54-191-44-4.us-west-2.compute.amazonaws.com/predict'

      url = URI.parse("#{url_base}?num=#{num}&rater_id=#{rater_id}&db_name=#{db_name}&db_host=#{db_host}&db_username=#{db_username}&db_password=#{db_password}&local=#{local}")
      req = Net::HTTP::Get.new(url.to_s)
      result = Net::HTTP.start(url.host, url.port) {|http|
        http.request(req)
      }
      result = JSON.parse(result.body)
    end

    # Disregard any lines before the last line
    result = result.split("\n").last
    # Parse joke ids
    result.split(' ').map{|s| s.to_i}
  end
end
