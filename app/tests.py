from django.test import TestCase
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class LoginTestCase(TestCase):
    def setUp(self):
        # Khởi tạo trình duyệt Selenium (đảm bảo bạn đã cài đặt driver cho trình duyệt bạn muốn sử dụng)
        self.driver = webdriver.Chrome()

    def tearDown(self):
        # Đóng trình duyệt khi kiểm thử hoàn thành
        self.driver.close()

    def test_login(self):
        # Mở trang đăng nhập
        self.driver.get(
            "http://localhost:8000/login/"
        )  # Thay đổi đường dẫn tùy theo ứng dụng của bạn

        # Nhập thông tin đăng nhập và bấm nút đăng nhập
        email_input = self.driver.find_element_by_name("email")
        password_input = self.driver.find_element_by_name("password")

        email_input.send_keys("tathiduyen73@gmail.com")
        password_input.send_keys("04022002")
        password_input.send_keys(Keys.RETURN)

        # Kiểm tra xem người dùng đã đăng nhập thành công hay không
        welcome_message = self.driver.find_element_by_id(
            "welcome-message"
        )  # Thay đổi id tùy theo ứng dụng của bạn
        self.assertEqual(welcome_message.text, "Welcome, tathiduyem73@gmail.com")


if __name__ == "__main__":
    import unittest

    unittest.main()
