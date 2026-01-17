import pandas as pd
import requests
import time


def get_pga_stats(stat_id, season_year):
    """
    Scrapes PGA Tour stats using their public hidden JSON or HTML tables.
    Note: '02564' is usually SG: Tee-to-Green, '02675' is SG: Total, etc.
    For simplicity, we will scrape ESPN's accessible tables which are easier.
    """
    # URL structure for ESPN Golf Stats (Standardized)
    # 2024 Season: https://www.espn.com/golf/stats/player/_/season/2024
    url = f"https://www.espn.com/golf/stats/player/_/season/{season_year}/table/general/sort/cupPoints/dir/desc"

    try:
        # Pandas can read HTML tables directly!
        # We use a header to look like a real browser
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)

        # Pull all tables from the page
        tables = pd.read_html(response.content)

        # ESPN usually splits data into two tables (Rank/Name and Stats). We merge them.
        df_names = tables[0]
        df_stats = tables[1]

        # Merge on index since they align perfectly on the page
        full_df = pd.concat([df_names, df_stats], axis=1)

        return full_df

    except Exception as e:
        print(f"Error scraping {season_year}: {e}")
        return pd.DataFrame()


# --- EXECUTION ---
print("Scraping 2024 Data...")
df_2024 = get_pga_stats('general', 2024)

print("Scraping 2025 Data...")
df_2025 = get_pga_stats('general', 2025)

# Clean and Combine
df_2024['Season'] = 2024
df_2025['Season'] = 2025
final_df = pd.concat([df_2024, df_2025])

# View columns to see what we got
print(final_df.columns)

# Save to CSV for your Model to use
final_df.to_csv("golf_stats_history.csv", index=False)
print("Data saved to golf_stats_history.csv")