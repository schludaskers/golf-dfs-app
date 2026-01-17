import time
import pandas as pd
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def get_full_pga_stats(season_year):
    print(f"\n--- Starting Scrape for {season_year} ---")

    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Comment this out to SEE the browser working!
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = f"https://www.espn.com/golf/stats/player/_/season/{season_year}/table/general/sort/cupPoints/dir/desc"
    driver.get(url)

    # Wait for initial load
    time.sleep(5)

    click_count = 0
    max_clicks = 10  # Safety limit

    while click_count < max_clicks:
        try:
            # Wait up to 10 seconds for the "Show More" link to be clickable
            wait = WebDriverWait(driver, 10)

            # ESPN usually uses an anchor tag <a> with text "Show More"
            # We use a CSS selector that finds ANY <a> tag containing that text
            # This is more robust than exact matching
            show_more_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Show More')]")))

            # Scroll it into view (sometimes it hides behind footers)
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_more_btn)
            time.sleep(1)

            # JavaScript click is often more reliable than standard .click() for these buttons
            driver.execute_script("arguments[0].click();", show_more_btn)

            click_count += 1
            print(f"Clicked 'Show More' ({click_count})...")

            # Wait for rows to populate
            time.sleep(4)

        except Exception as e:
            # If we timeout, it likely means the button is gone (we reached the end)
            print("No more 'Show More' buttons found (or end of list).")
            break

    # 3. Pass the fully loaded HTML to Pandas
    try:
        html = driver.page_source
        tables = pd.read_html(StringIO(html))

        if len(tables) >= 2:
            # Table 0 = Rk/Name, Table 1 = Stats
            full_df = pd.concat([tables[0], tables[1]], axis=1)
        else:
            full_df = tables[0]

        print(f"Found {len(full_df)} players for {season_year}.")
        driver.quit()
        return full_df

    except Exception as e:
        print(f"Error parsing table: {e}")
        driver.quit()
        return pd.DataFrame()


# --- EXECUTION ---
df_2024 = get_full_pga_stats(2024)
df_2025 = get_full_pga_stats(2025)

if not df_2024.empty: df_2024['Season'] = 2024
if not df_2025.empty: df_2025['Season'] = 2025

final_df = pd.concat([df_2024, df_2025], ignore_index=True)

if not final_df.empty:
    final_df.to_csv("golf_stats_history.csv", index=False)
    print(f"\nSUCCESS! Saved {len(final_df)} total rows to 'golf_stats_history.csv'")
else:
    print("Scrape failed.")