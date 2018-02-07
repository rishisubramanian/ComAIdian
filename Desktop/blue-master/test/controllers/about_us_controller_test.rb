require 'test_helper'

#Only need to test that the actual page is up as its static html

class AboutUsControllerTest < ActionDispatch::IntegrationTest
  test "should get index" do
    about_us_index_url = "http://localhost:3000/about_us"
    get about_us_index_url
    #puts response.body
    assert_response :success
  end

end
