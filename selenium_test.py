from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
import time

# Initialize WebDriver (e.g., Chrome)
driver = webdriver.Chrome()

# Apply stealth settings to avoid detection
stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
        )

# Navigate to Tinder's website
driver.get("https://tinder.com")

# Allow the page to load
time.sleep(5)

# Locate and click the 'Log in' button by class name and text content
try:
    login_button = driver.find_element(By.XPATH, "//div[@class='lxn9zzn' and text()='Log in']")
    login_button.click()
    print("Login button clicked.")
except Exception as e:
    print(f"Error: {e}")

# Add additional steps here to handle the login process if needed

# Close the browser after operations
time.sleep(5)  # Adjust sleep time as needed
driver.quit()
