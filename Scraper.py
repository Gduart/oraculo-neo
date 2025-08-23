# scraper.py
import sys
import os
from playwright.sync_api import sync_playwright
from fake_useragent import UserAgent

def scrape_website(url: str, output_file: str):
    """
    Usa Playwright para navegar até uma página, aguarda o conteúdo ser renderizado
    e salva o texto do corpo da página em um arquivo de saída.
    """
    try:
        with sync_playwright() as p:
            ua = UserAgent()
            user_agent_string = ua.random
           
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=user_agent_string)
            page = context.new_page()
           
            page.goto(url, timeout=90000, wait_until="domcontentloaded")
            page.wait_for_timeout(7000) # Espera crucial para renderização de JS

            content = page.locator('body').inner_text()
            browser.close()

            if not content or not content.strip():
                raise ValueError("Nenhum conteúdo de texto foi encontrado no corpo da página.")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
           
    except Exception as e:
        # Salva a mensagem de erro no arquivo de saída para o script principal ler
        error_message = f"SCRAPER_ERROR: {str(e)}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(error_message)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        url_to_scrape = sys.argv[1]
        output_filename = sys.argv[2]
        scrape_website(url_to_scrape, output_filename)
