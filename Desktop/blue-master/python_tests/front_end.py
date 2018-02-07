import unittest
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class ComedAIn(unittest.TestCase):
    remoteURL = "http://ec2-18-216-215-8.us-east-2.compute.amazonaws.com:4000/"
    localURL = "http://0.0.0.0:4000/"

    def setUp(self):
        self.driver = webdriver.Chrome()
        self.base_url = ComedAIn.localURL

    # Place holder test to show that testing program works
    def test_placeholder(self):
        driver = self.driver
        # driver.get(self.base_url + "#/information")
        driver.get("http://python.org")
        
        self.assertIn("Python", driver.title)
        elem = driver.find_element_by_name("q")
        elem.send_keys("pycon")
        elem.send_keys(Keys.RETURN)
        assert "No results found." not in driver.page_source


    def tearDown(self):
        self.driver.close()

if __name__ == "__main__":
    unittest.main()