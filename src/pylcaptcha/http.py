import time
from curl_cffi import AsyncSession, Response
from pylcaptcha.captcha import Captcha
from pylcaptcha.playwright_wrapper import Browser
from playwright.async_api import Response as pw_Response
class CloudFlareBypassError(Exception):
    pass


class CaptchaBypassError(Exception):
    pass


class BrowserHTTP:
    def __init__(self, bypass_on_expire=False):
        self.session = AsyncSession(impersonate='firefox135')
        self.browser = Browser()
        self.bypass_on_expire = bypass_on_expire

    async def __aenter__(self):
        """Triggers when entering the 'async with' block."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Triggers automatically when exiting the block, even if an error occurs."""
        await self.browser.close()
    async def _handle_cloudflare_challenge(self, url: str) -> bool:
        await self.browser.GET(url)

        try:
            captcha = Captcha(self.browser.page)
            token = await captcha.solve_interstitial_cf_captcha()
            token2 = await captcha.solve_embedded_cf_captcha()
            if token2:
                token = token2
        except Exception as e:
            raise CaptchaBypassError(f"The solver encountered an unhandled exception: {e}")

        if not token or token in ('interstitial_failed', 'failed'):
            return False

        # Save the Turnstile token internally
        self.cf_token = token

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
        return True

    async def _request(self, method: str, url: str, browser=False, **kwargs) -> Response | pw_Response:
        """
        Central router that manages payloads, detects blocks, and triggers retries.
        """
        if await self.check_cf_token_expired() and self.bypass_on_expire or browser:
            response = await self._handle_cloudflare_challenge(url)
            if not response:
                raise CaptchaBypassError("Solver ran but failed to return a valid clearance token.")
            return response
        # Execute the primary request (preserves headers, json, data, params, etc.)
        result = await getattr(self.session, method.lower())(url, **kwargs)

        if result.status_code in (403, 503):
            # Trigger the bypass flow
            solved = await self._handle_cloudflare_challenge(url)

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
    async def get_cf_token(self) -> str:
        return self.cf_token or ""
    # Standard public entrypoints
    async def get(self, url: str, browser=False, **kwargs) -> Response:
        return await self._request('GET', url, browser, **kwargs)

    async def post(self, url: str, browser=False, guess_csrf: bool = True, **kwargs) -> Response:
        return await self._request('POST', url, browser, **kwargs)