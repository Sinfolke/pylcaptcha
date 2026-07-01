import time
from curl_cffi import AsyncSession, Response
from playwright.async_api import Response as pw_Response
from pydantic import BaseModel
from bs4 import BeautifulSoup

from pylcaptcha.captcha import Captcha
from pylcaptcha.playwright_wrapper import Browser
class CloudFlareBypassError(Exception):
    pass

class CaptchaBypassError(Exception):
    pass

class Options(BaseModel):
    cf_auto_bypass: bool = True
    cf_embedded_captcha_auto_bypass: bool = True
    cf_bypass_on_expire: bool = True
    g_auto_token_track: bool = True
    browser_headless_mode: bool = False
class BrowserHTTP:
    def __init__(self, options: Options = Options(), proxy: dict = None):
        self.options = options

        proxy_str = None

        if proxy:
            server = proxy.get("server")

            # Camoufox format: username/password separated
            username = proxy.get("username")
            password = proxy.get("password")

            if username and password:
                # insert credentials into URL
                proto, rest = server.split("://", 1)
                proxy_str = f"{proto}://{username}:{password}@{rest}"
            else:
                proxy_str = server

        self.session = AsyncSession(
            impersonate="firefox135",
            proxies=proxy_str
        )

        self.browser = Browser(proxy)
        self.cf_token = None
        self.gc_token = None
    async def __aenter__(self):
        """Triggers when entering the 'async with' block."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Triggers automatically when exiting the block, even if an error occurs."""
        await self.browser.close()
    async def _handle_challenge(self, url: str, method: str, **kwargs):
        print(f"executing {method} on {url}, getattr: {getattr(self.browser, method)}")
        response = await getattr(self.browser, method)(url, **kwargs, headless=self.options.browser_headless_mode)
        if self.options.cf_auto_bypass:
            try:
                captcha = Captcha(self.browser.page)
                token = await captcha.solve_interstitial_cf_captcha()
                if self.options.cf_embedded_captcha_auto_bypass:
                    token2 = await captcha.solve_embedded_cf_captcha()
                    if token2:
                        token = token2
            except Exception as e:
                raise CaptchaBypassError(f"The solver encountered an unhandled exception: {e}")

            if not token or token in ('interstitial_failed', 'failed'):
                raise CaptchaBypassError("Could not extract token from solved captcha")

            # Save the Turnstile token internally
            self.cf_token = token
        if self.options.g_auto_token_track:
            captcha = Captcha(self.browser.page)
            if await captcha.check_g_captcha():
                self.gc_token = await captcha.solve_g_captcha()


        # Sync browser identity to curl_cffi session
        browser_user_agent = await self.browser.page.evaluate("navigator.userAgent")
        self.session.headers.update({
            "User-Agent": browser_user_agent,
            "x-requested-with": "XMLHttpRequest"
        })

        # Sync cookie context jars
        browser_cookies = await self.browser.page.context.cookies(urls=url)
        for cookie in browser_cookies:
            self.session.cookies.set(
                name=cookie['name'],
                value=cookie['value'],
                domain=cookie['domain'],
                path=cookie['path']
            )
        return response
    async def _request(self, method: str, url: str, browser=False, **kwargs) -> Response | pw_Response:
        """
        Central router that manages payloads, detects blocks, and triggers retries.
        """
        if browser:
            response = await self._handle_challenge(url, method, **kwargs)
            if not response:
                raise CaptchaBypassError("Solver ran but failed to return a valid clearance token.")
            return response
        # Execute the primary request (preserves headers, JSON, data, params, etc.)
        result = await getattr(self.session, method.lower())(url, **kwargs)

        if result.status_code in (403, 503):
            # Trigger the bypass flow
            solved = await self._handle_challenge(url, method)

            if not solved:
                raise CaptchaBypassError("Solver ran but failed to return a valid clearance token.")

            # Retry the exact same request with original payloads preserved
            result = await getattr(self.session, method.lower())(url, **kwargs)

            # If it STILL fails after synchronization, Cloudflare has rejected our fingerprint
            if result.status_code in (403, 503):
                raise CloudFlareBypassError("Session was blocked again immediately after applying fresh cookies.")
        return result

    async def check_cf_token_expired(self) -> bool:
        """
        Checks if the cf_clearance cookie exists and is still valid.
        """
        import time
        now = time.time()
        cf_cookie = None

        # Fix: Step into the underlying standard library CookieJar
        for cookie in self.session.cookies.jar:
            if cookie.name == 'cf_clearance':
                cf_cookie = cookie
                break

        if not cf_cookie:
            return True  # No token means it is effectively expired/missing

        # If the cookie has an explicit expiration timestamp, validate it
        if cookie.expires and cookie.expires < now:
            return True

        return False

    async def sync_csrf_token(self, selector: str, header_name: str = None, cookie_name: str = None) -> str:
        """
        Extracts a token from the browser DOM using a generic selector,
        safely handling quotes via Playwright argument casting.
        """
        # Fix: Dropped the 'f' prefix and passed 'selector' as a native trailing argument
        extracted_token = await self.browser.page.evaluate("""
                                                           (targetSelector) => {
                                                               const el = document.querySelector(targetSelector);
                                                               if (!el) return "";
                                                               return el.value || el.content || el.innerText || "";
                                                           }
                                                           """, selector)

        if not extracted_token:
            return ""

        # Map to custom header if provided
        if header_name:
            self.session.headers.update({header_name: extracted_token})

        # Map to custom cookie jar if provided
        if cookie_name:
            from urllib.parse import urlparse
            current_url = self.browser.page.url
            domain = f".{urlparse(current_url).netloc.replace('www.', '')}"
            self.session.cookies.set(name=cookie_name, value=extracted_token, domain=domain, path='/')

        return extracted_token
    def sync_csrf_token_from_raw(self, html_content: str, selector: str, header_name: str = None,
                                 cookie_name: str = None) -> str:
        """
        Extracts a token from raw HTML text using a CSS selector (e.g., 'meta[name="csrf-token"]').
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Locate the element using the CSS selector
        el = soup.select_one(selector)

        if not el:
            return ""

        # 2. Extract the value from standard attributes or text content
        extracted_token = el.get("value") or el.get("content") or el.get("text") or el.string or ""

        extracted_token = extracted_token.strip()
        if not extracted_token:
            return ""

        # 3. Map to custom header if provided
        if header_name:
            self.session.headers.update({header_name: extracted_token})

        # 4. Map to custom cookie jar if provided
        if cookie_name:
            from urllib.parse import urlparse
            # Fallback to the session's base URL or pass the current target URL as an argument
            current_url = self.session.url if hasattr(self.session, "url") else "https://imei24.com"
            domain = f".{urlparse(current_url).netloc.replace('www.', '')}"
            self.session.cookies.set(name=cookie_name, value=extracted_token, domain=domain, path='/')

        return extracted_token
    async def get_cf_token(self) -> str:
        return self.cf_token or ""
    async def get_gc_token(self):
        return self.gc_token or ""
    # Standard public entrypoints
    async def get(self, url: str, browser=False, **kwargs) -> Response:
        return await self._request('GET', url, browser, **kwargs)

    async def post(self, url: str, browser=False, guess_g_captcha: bool = True,
                   **kwargs) -> Response:
        """
        Standard POST entrypoint with optional smart CAPTCHA token injection.
        """
        if guess_g_captcha and self.gc_token:
            # 1. Handle Form Data (application/x-www-form-urlencoded)
            if 'data' in kwargs:
                if isinstance(kwargs['data'], dict):
                    kwargs['data']['g-recaptcha-response'] = self.gc_token
                elif isinstance(kwargs['data'], str) and "g-recaptcha-response=" not in kwargs['data']:
                    # Append to raw string payload if it's string-encoded form data
                    separator = "&" if kwargs['data'] else ""
                    kwargs['data'] += f"{separator}g-recaptcha-response={self.gc_token}"

            # 2. Handle JSON Payloads (application/json)
            elif 'json' in kwargs and isinstance(kwargs['json'], dict):
                # We inject the standard key, but the user can override it manually in kwargs if the API uses a custom key
                if 'g-recaptcha-response' not in kwargs['json']:
                    kwargs['json']['g-recaptcha-response'] = self.gc_token

            # 3. Fallback: If no payload was provided, initialize form data with the token
            elif 'data' not in kwargs and 'json' not in kwargs:
                kwargs['data'] = {'g-recaptcha-response': self.gc_token}
            self.gc_token = None
        return await self._request('POST', url, browser, **kwargs)
    async def close(self):
        await self.browser.close()