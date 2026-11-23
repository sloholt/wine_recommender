"""
Reusable wine report scraper.

Extracts:
    - Rank
    - Title
    - Score
    - Color
    - Country
    - Region
    - Vintage
    - Avg. Price (when shown)
    - Public tasting-note preview
    - Source URL

Run:
    python -m ingestion.scraper \
        --url "https://example.com/report" \
        --country chile \
        --year 2026 \
        --pages 5

"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import re
import time

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    TimeoutException,
    StaleElementReferenceException,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "top100json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "WineResearchBot/1.0"
REQUEST_TIMEOUT = 20
REQUEST_DELAY = 3

WINE_CARD_SELECTOR = "#tasting-note-list div.bg-gray-100.rounded-xl.mb-2"

# Placeholder: confirm from pagination HTML
NEXT_PAGE_SELECTOR = 'a[rel="next"]'

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_robots(url):
    """
    Check whether robots.txt permits crawling the URL.
    """

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        response = requests.get(
            robots_url, headers={"User-Agent": USER_AGENT}, timeout=15
        )

        response.raise_for_status()

        parser = RobotFileParser()
        parser.parse(response.text.splitlines())

        allowed = parser.can_fetch(USER_AGENT, url)

        print(f"Robots URL: {robots_url}")
        print(f"Robots HTTP status: {response.status_code}")
        print(f"Crawling permitted by robots.txt: {allowed}")

        return allowed

    except requests.RequestException as e:
        logger.error(f"Unable to verify robots.txt: {e}")
        return False


def fetch_html(url, use_selenium=False):
    """
    Fetch a public page.

    Does not log in or bypass access controls.
    """

    if not check_robots(url):
        raise PermissionError(f"Automated access not confirmed: {url}")

    if use_selenium:

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1400,1000")

        driver = webdriver.Chrome(options=options)

        try:
            driver.get(url)
            time.sleep(3)
            return driver.page_source
        finally:
            driver.quit()

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code in (401, 403, 429):
        raise PermissionError(f"Access restricted: HTTP {response.status_code}")

    response.raise_for_status()

    return response.text


# EXTRACTION HELPERS


def clean_text(value):
    if not value:
        return None

    return " ".join(value.split())


def extract_text(element, selector):
    match = element.select_one(selector)

    if match is None:
        return None

    return clean_text(match.get_text(" ", strip=True))


def extract_field(text, field):
    """
    Extract a labeled field from multiline text.
    """

    pattern = rf"(?im)^{re.escape(field)}\s*:?\s*(.+)$"

    match = re.search(pattern, text)

    return clean_text(match.group(1)) if match else None


def parse_number(value, min_value, max_value):
    if not value:
        return None

    match = re.search(r"\d+(?:\.\d+)?", value)

    if not match:
        return None

    number = float(match.group())

    if min_value <= number <= max_value:
        return int(number) if number.is_integer() else number

    return None


CURRENCY_SYMBOLS = {
    "US$": "USD",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "HK$": "HKD",
    "A$": "AUD",
    "C$": "CAD",
}


def parse_price(value):
    """
    Parse an 'Avg. Price' value such as '$45', 'US$ 1,200', '€30' or '45 USD'.

    Returns (price, currency) or (None, None). Requires a digit, so a missing
    value that spills onto the next label line (e.g. 'Score') is ignored.
    """

    if not value:
        return None, None

    match = re.search(r"\d[\d,]*(?:\.\d+)?", value)

    if not match:
        return None, None

    price = float(match.group().replace(",", ""))

    currency = None
    code = re.search(r"\b([A-Z]{3})\b", value)

    if code:
        currency = code.group(1)
    else:
        # Longest symbols first so 'US$' wins over '$'
        for symbol in sorted(CURRENCY_SYMBOLS, key=len, reverse=True):
            if symbol in value:
                currency = CURRENCY_SYMBOLS[symbol]
                break

    return (int(price) if price.is_integer() else price), currency


# WINE TITLE AND RANK


def extract_heading(block):
    """
    Extract rank and wine title from card button.
    """

    button = block.select_one("button")

    if button is None:
        return None, None

    heading = clean_text(button.get_text(" ", strip=True))

    if not heading:
        return None, None

    match = re.match(r"^(\d+)\s+(.+)$", heading)

    if match:
        return int(match.group(1)), match.group(2)

    return None, heading


# TASTING DESCRIPTION


def extract_tasting_note(block):
    """
    Extract a publicly visible tasting-note preview.

    Do not attempt to reveal subscriber-only content.
    """

    panel = block.select_one('[id^="headlessui-disclosure-panel"]')

    if panel is None:
        return None

    # Provisional selector based on the supplied HTML.
    note_container = panel.select_one("div.desktop\\:col-start-5")

    if note_container is None:
        return None

    paragraphs = note_container.select("p")

    if paragraphs:
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    else:
        text = note_container.get_text(" ", strip=True)

    return clean_text(text)


# PARSE INDIVIDUAL WINE
# Is the element actually the topmost thing at its own center point?
IS_UNCOVERED_JS = """
const el = arguments[0];
const r = el.getBoundingClientRect();
const x = r.left + r.width / 2, y = r.top + r.height / 2;
if (y < 0 || y > window.innerHeight) return false;
const top = document.elementFromPoint(x, y);
return top === el || el.contains(top);
"""

# Describe whatever is covering the element (for debugging)
BLOCKER_JS = """
const r = arguments[0].getBoundingClientRect();
const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
return top ? top.outerHTML.slice(0, 200) : null;
"""


def expand_wine_cards(driver):
    """
    Expand wine cards using their visible disclosure buttons.
    """
    wait = WebDriverWait(driver, 15)
    short_wait = WebDriverWait(driver, 3)

    count = len(driver.find_elements(By.CSS_SELECTOR, WINE_CARD_SELECTOR))
    logger.info("Found %d wine cards to expand", count)

    for index in range(count):

        # Always re-find: expanding a card re-renders the list
        def get_button(d=driver):
            cards = d.find_elements(By.CSS_SELECTOR, WINE_CARD_SELECTOR)
            return cards[index].find_element(By.CSS_SELECTOR, "button")

        if get_button().get_attribute("aria-expanded") == "true":
            continue

        # Instant scroll, so the site's smooth scrolling can't leave it mid-animation
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});",
            get_button(),
        )

        try:
            # Wait until nothing is covering the button, then click normally
            short_wait.until(lambda d: d.execute_script(IS_UNCOVERED_JS, get_button(d)))
            get_button().click()

        except (TimeoutException, ElementClickInterceptedException):
            blocker = driver.execute_script(BLOCKER_JS, get_button())
            logger.warning(
                "Card %d is covered by: %s -- using JS click", index + 1, blocker
            )
            driver.execute_script("arguments[0].click();", get_button())

        # Wait for the disclosure panel to open
        wait.until(lambda d: get_button(d).get_attribute("aria-expanded") == "true")

        logger.info("Expanded wine card %d/%d", index + 1, count)


def parse_wine_card(block, source_url):
    """
    Parse one wine result card.
    """

    rank, title = extract_heading(block)

    panel = block.select_one('[id^="headlessui-disclosure-panel"]')

    details_text = panel.get_text("\n", strip=True) if panel else ""

    score = parse_number(extract_field(details_text, "Score"), 0, 100)

    vintage = parse_number(extract_field(details_text, "Vintage"), 1800, 2100)

    tasting_note = extract_tasting_note(block)

    # 'Avg. Price' badge (e.g. "$ 118") is inside the disclosure panel
    price, currency = parse_price(extract_field(details_text, "Avg. Price"))

    return {
        "rank": rank,
        "title": title,
        "winery": None,
        "varietal": None,
        "score": score,
        "color": extract_field(details_text, "Color"),
        "country": extract_field(details_text, "Country"),
        "region": extract_field(details_text, "Region"),
        "vintage": vintage or extract_vintage(title),
        "price": price,
        "currency": currency,
        "tasting_note": tasting_note,
        "tasting_note_truncated": (
            tasting_note is not None and tasting_note.endswith(("...", "…"))
        ),
        "source_name": "James Suckling",
        "source_url": source_url,
    }


# PARSE REPORT PAGE
def extract_vintage(title):
    match = re.search(r"\b(19|20)\d{2}\b$", title or "")
    return int(match.group()) if match else None


def parse_page(html, source_url):
    """
    Extract all wine cards from a report page.
    """

    soup = BeautifulSoup(html, "lxml")

    cards = soup.select(WINE_CARD_SELECTOR)

    logger.info("Found %d wine blocks", len(cards))

    wines = []

    for card in cards:
        wine = parse_wine_card(card, source_url)

        if wine["title"]:
            wines.append(wine)

    next_link = soup.select_one(NEXT_PAGE_SELECTOR)

    next_url = None

    if next_link and next_link.get("href"):
        next_url = urljoin(source_url, next_link["href"])

    return wines, next_url


def go_to_page(driver, page_number):
    """
    Navigate to a numbered page and wait for
    the wine listings to update.
    """

    wait = WebDriverWait(driver, 20)

    old_content = driver.find_element(By.ID, "tasting-note-list").text

    xpath = f'//a[span[normalize-space()="{page_number}"]]'

    # 1. Scroll to the bottom of the webpage
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

    # 2. Wait for the layout to settle
    time.sleep(1)

    # 3. Find the pagination button after scrolling
    page_link = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))

    # 4. Print diagnostic information
    print(
        driver.execute_script(
            """
        return {
            innerHeight: window.innerHeight,
            innerWidth: window.innerWidth,
            scrollY: window.scrollY,
            documentHeight: document.documentElement.scrollHeight,
            buttonRect: arguments[0].getBoundingClientRect().toJSON()
        };
    """,
            page_link,
        )
    )

    # 5. Click the next page
    page_link.click()

    # 6. Wait for the wine listings to change
    wait.until(lambda d: d.find_element(By.ID, "tasting-note-list").text != old_content)
    # Wait until the new page contains wine cards
    wait.until(
        lambda d: len(d.find_elements(By.CSS_SELECTOR, WINE_CARD_SELECTOR)) >= 20
    )
    print(f"Successfully navigated to page {page_number}")


# MULTI-PAGE SCRAPER
def scrape_report(start_url, max_pages=5, use_selenium=True):
    """
    Scrape multiple pages using one Selenium session.
    """

    if not check_robots(start_url):
        logger.error(
            "Cannot proceed: robots.txt either disallows "
            "this URL or could not be verified."
        )
        return []

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")

    driver = webdriver.Chrome(options=options)

    all_wines = []
    seen_wines = set()

    try:
        driver.get(start_url)

        WebDriverWait(driver, 20).until(
            lambda d: d.find_elements(By.CSS_SELECTOR, WINE_CARD_SELECTOR)
        )

        for page_number in range(1, max_pages + 1):

            logger.info("Scraping page %d/%d", page_number, max_pages)
            logger.info("Expanding wine cards...")
            expand_wine_cards(driver)

            # Use your existing BeautifulSoup parser
            wines, _ = parse_page(driver.page_source, driver.current_url)

            for wine in wines:
                key = (wine["title"], wine["vintage"])

                if key not in seen_wines:
                    seen_wines.add(key)
                    all_wines.append(wine)

            logger.info(
                "Page %d: %d wines | Total: %d", page_number, len(wines), len(all_wines)
            )

            if page_number < max_pages:
                delay = random.uniform(3, 6)

                logger.info("Waiting %.1f seconds before next page...", delay)

                time.sleep(delay)

                go_to_page(driver, page_number + 1)

    finally:
        driver.quit()

    expected_wines = max_pages * 20

    if len(all_wines) != expected_wines:
        logger.warning(
            "Expected %d wines, extracted %d", expected_wines, len(all_wines)
        )

    return all_wines


# JSON EXPORT


def save_json(wines, country, year):
    """
    Save one JSON file per country/year.
    """

    filename = f"{year}_top100_{country.lower()}.json"

    output_path = OUTPUT_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(wines, file, indent=2, ensure_ascii=False)

    logger.info("Saved %d wines to %s", len(wines), output_path)

    return output_path


def main():

    parser = argparse.ArgumentParser(description="Scrape wine report listings")

    parser.add_argument("--url", required=True, help="Report URL")

    parser.add_argument("--country", required=True, help="Country name")

    parser.add_argument("--year", required=True, type=int, help="Report year")

    parser.add_argument("--pages", type=int, default=5, help="Maximum pages")

    parser.add_argument(
        "--selenium", action="store_true", help="Render JavaScript using Selenium"
    )

    args = parser.parse_args()

    wines = scrape_report(
        start_url=args.url, max_pages=args.pages, use_selenium=args.selenium
    )

    if not wines:
        logger.warning(
            "No wine records extracted. " "Verify permissions, selectors, and HTML."
        )
        return

    save_json(wines, country=args.country, year=args.year)

    print(f"\nSuccessfully extracted {len(wines)} wines.")


if __name__ == "__main__":
    main()
