from __future__ import annotations

import asyncio
from dataclasses import dataclass
from http.cookiejar import CookieJar
from urllib.parse import urlencode, urlparse
from typing import Optional, List, Any

from camoufox import DefaultAddons
from playwright.async_api import Page, Route, Locator, async_playwright, ViewportSize, BrowserContext
from camoufox.async_api import AsyncCamoufox

class Browser:
    def __init__(self, proxy: dict):
        self.url: Optional[str] = None
        self.data: Optional[dict] = None
        self.camoufox_wrapper = None
        self.browser = None
        self.context = None
        self.page: Optional[Page] = None
        self.proxy = proxy

    async def _inject_cookies(self, cookie_jar: CookieJar, domain: str):
        playwright_cookies = []
        for cookie in cookie_jar:
            p_cookie = {
                "name": cookie.name,
                "value": cookie.value,
                "domain": domain,
                "path": cookie.path or "/",
            }
            if cookie.expires:
                p_cookie["expires"] = cookie.expires
            playwright_cookies.append(p_cookie)

        await self.context.add_cookies(playwright_cookies)

    async def _enable_ad_blocker(self):
        blocked_patterns = [
            "*://*.doubleclick.net/*",
            "*://*.googlesyndication.com/*",
            "*://*.googleadservices.com/*",
            "*://*.googletagmanager.com/*",
            "*://*.facebook.net/*",
            "*://*.facebook.com/tr/*",
            "*://*.analytics.google.com/*",
            "*://*.adnxs.com/*",
        ]

        # 1. Block known tracker/ad domains
        for pattern in blocked_patterns:
            await self.context.route(pattern, lambda route: route.abort())

        # 2. Heuristic and resource blocking
        async def block_resources(route: Route):
            req = route.request
            url = req.url.lower()
            rtype = req.resource_type

            if any(x in url for x in ["ads", "adservice", "doubleclick", "banner"]):
                return await route.abort()

            if rtype in ["font", "media"]:
                return await route.abort()

            return await route.continue_()

        await self.context.route("**/*", block_resources)

    async def _setup_browser(self, cookie=None, block_adv=False, headless=True, Camoufox=True, PersistentContext = True, Proxy: dict | None = None):
        if self.page is not None:
            return
        if not Proxy and self.proxy:
            Proxy = self.proxy
        self.camoufox_wrapper = AsyncCamoufox(
            persistent_context=PersistentContext,
            user_data_dir="./camoufox_session_data" if PersistentContext else None,
            locale="en-US",
            headless=headless,
            humanize=False,
            geoip=True,
            exclude_addons=[DefaultAddons.UBO],
            os = "windows",
            proxy=Proxy
        )
        self.context = await self.camoufox_wrapper.__aenter__()
        if len(self.context.pages) > 0:
            self.page = self.context.pages[0]
        else:
            self.page = await self.context.new_page()
        if block_adv:
            await self._enable_ad_blocker()

        if cookie and self.url:
            domain = urlparse(self.url).hostname
            if domain:
                await self._inject_cookies(cookie, domain)

    async def GET(self, url: str, data: Optional[dict[str, str]] = None, cookie: Optional[CookieJar] = None,
                  block_adv: bool = False, headless: bool = True, CamouFox: bool = True, PersistentContext: bool = True, proxy: dict[str, str] | None = None):
        self.url = f"{url}?{urlencode(data)}" if data else url
        await self._setup_browser(cookie, block_adv, headless, CamouFox, PersistentContext, proxy)

        result = await self.page.goto(self.url, wait_until="domcontentloaded")
        return result

    async def POST(self, url: str, data: Optional[dict[str, str]] = None, cookie: Optional[CookieJar] = None, block_adv: bool = False,
                   headless: bool = True, CamouFox: bool = True, PersistentContext: bool = True, proxy: dict[str, str] | None = None):
        self.url = url
        self.data = data
        await self._setup_browser(cookie, block_adv, headless, CamouFox, PersistentContext, proxy)

        async def handle_route(route: Route):
            if route.request.url == self.url:
                # Transform the outgoing request into a POST
                await route.continue_(
                    method="POST",
                    post_data=urlencode(self.data),
                    headers={
                        **route.request.headers,
                        "Content-Type": "application/x-www-form-urlencoded"
                    }
                )
            else:
                await route.continue_()

        # Route only the specific endpoint URL, then remove the route after execution
        await self.context.route(self.url, handle_route)
        result = await self.page.goto(self.url, wait_until="domcontentloaded")
        await self.context.unroute(self.url, handle_route)

        return result

    async def close(self):
        """Ensure clean teardown of Playwright and Camoufox binaries."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
        finally:
            if self.camoufox_wrapper:
                await self.camoufox_wrapper.__aexit__(None, None, None)


@dataclass
class ClassList:
    node: "DOMNode"

    async def add(self, class_name: str) -> None:
        await self.node.evaluate("(el, cls) => el.classList.add(cls)", class_name)

    async def remove(self, class_name: str) -> None:
        await self.node.evaluate("(el, cls) => el.classList.remove(cls)", class_name)

    async def contains(self, class_name: str) -> bool:
        return await self.node.evaluate(
            "(el, cls) => el.classList.contains(cls)", class_name
        )
    def _css_escape(value: str) -> str:
        return value.replace('"', '\\"')

@dataclass
class DOMNodeList:
    locator: Locator

    async def count(self) -> int:
        return await self.locator.count()

    def first(self) -> "DOMNode":
        return DOMNode(self.locator.first)

    def last(self) -> "DOMNode":
        return DOMNode(self.locator.last)

    def nth(self, index: int) -> "DOMNode":
        return DOMNode(self.locator.nth(index))

    async def all(self) -> List["DOMNode"]:
        total = await self.locator.count()
        return [DOMNode(self.locator.nth(i)) for i in range(total)]

    async def clickAll(self, **kwargs: Any) -> None:
        total = await self.locator.count()
        for i in range(total):
            await self.locator.nth(i).click(**kwargs)

    async def innerTexts(self) -> List[str]:
        total = await self.locator.count()
        return [await self.locator.nth(i).inner_text() for i in range(total)]

    def __getitem__(self, index: int) -> "DOMNode":
        return self.nth(index)

@dataclass
class DOMNode:
    locator: Locator

    def querySelector(self, selector: str) -> "DOMNode":
        return DOMNode(self.locator.locator(selector).first)

    def querySelectorAll(self, selector: str) -> DOMNodeList:
        return DOMNodeList(self.locator.locator(selector))

    def getElementById(self, element_id: str) -> "DOMNode":
        return self.querySelector(f"#{element_id}")

    def getElementsByClassName(self, class_name: str) -> DOMNodeList:
        return self.querySelectorAll(f".{class_name}")

    def getElementsByName(self, name: str) -> DOMNodeList:
        return self.querySelectorAll(f'[name="{self._css_escape(name)}"]')

    def parentElement(self) -> "DOMNode":
        return DOMNode(self.locator.locator("xpath=..").first)

    def children(self) -> DOMNodeList:
        return DOMNodeList(self.locator.locator(":scope > *"))

    @property
    def classList(self) -> ClassList:
        return ClassList(self)

    async def exists(self) -> bool:
        return await self.locator.count() > 0

    async def count(self) -> int:
        return await self.locator.count()

    async def childElementCount(self) -> int:
        return await self.locator.locator(":scope > *").count()

    async def isVisible(self) -> bool:
        try:
            return await self.locator.first.is_visible()
        except Exception:
            return False

    async def isEnabled(self) -> bool:
        try:
            return await self.locator.first.is_enabled()
        except Exception:
            return False

    async def click(self, **kwargs: Any) -> None:
        await self.locator.first.click(**kwargs)

    async def dblclick(self, **kwargs: Any) -> None:
        await self.locator.first.dblclick(**kwargs)

    async def hover(self, **kwargs: Any) -> None:
        await self.locator.first.hover(**kwargs)

    async def fill(self, value: str, **kwargs: Any) -> None:
        await self.locator.first.fill(value, **kwargs)

    async def type(self, value: str, delay: float = 0) -> None:
        await self.locator.first.press_sequentially(value, delay=delay)

    async def check(self, **kwargs: Any) -> None:
        await self.locator.first.check(**kwargs)

    async def uncheck(self, **kwargs: Any) -> None:
        await self.locator.first.uncheck(**kwargs)

    async def innerText(self) -> str:
        return await self.locator.first.inner_text()

    async def innerHTML(self) -> str:
        return await self.locator.first.inner_html()

    async def textContent(self) -> Optional[str]:
        return await self.locator.first.text_content()

    async def inputValue(self) -> str:
        return await self.locator.first.input_value()

    async def getAttribute(self, name: str) -> Optional[str]:
        return await self.locator.get_attribute(name)

    async def setAttribute(self, name: str, value: str) -> None:
        await self.locator.first.evaluate(
            "(el, data) => el.setAttribute(data.name, data.value)",
            {"name": name, "value": value},
        )

    async def value(self):
        return await self.evaluate("el => el.value")
    async def removeAttribute(self, name: str) -> None:
        await self.locator.first.evaluate(
            "(el, attr) => el.removeAttribute(attr)",
            name,
        )

    async def waitVisible(self, timeout: float | None = None) -> None:
        await self.locator.first.wait_for(state="visible", timeout=timeout)

    async def waitHidden(self, timeout: float | None = None) -> None:
        await self.locator.first.wait_for(state="hidden", timeout=timeout)

    async def scrollIntoView(self, timeout: float | None = None) -> None:
        await self.locator.first.scroll_into_view_if_needed(timeout=timeout)

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        handle = await self.locator.first.element_handle()
        if handle is None:
            raise RuntimeError("Element not found")
        return await handle.evaluate(script, arg)

    async def safeClick(self, **kwargs: Any) -> None:
        await self.waitVisible()
        if not await self.isEnabled():
            raise RuntimeError("Element is disabled")
        await self.click(**kwargs)

    def _css_escape(value: str) -> str:
        return value.replace('"', '\\"')


class DOM:
    def __init__(self, root: RootType):
        self.root = root

    def querySelector(self, selector: str) -> DOMNode:
        return DOMNode(self.root.locator(selector).first)

    def querySelectorAll(self, selector: str) -> DOMNodeList:
        return DOMNodeList(self.root.locator(selector))

    def getElementById(self, element_id: str) -> DOMNode:
        return self.querySelector(f"#{element_id}")

    def getElementsByClassName(self, class_name: str) -> DOMNodeList:
        return self.querySelectorAll(f".{class_name}")

    def getElementsByName(self, name: str) -> DOMNodeList:
        return self.querySelectorAll(f'[name="{self._css_escape(name)}"]')

    def frame(self, selector: str) -> "DOM":
        frame_locator = getattr(self.root, "frame_locator", None)
        if frame_locator is None:
            raise TypeError(
                f"Current DOM root does not support frame(). Root type: {type(self.root)!r}"
            )
        return DOM(frame_locator(selector).first)
    def _css_escape(value: str) -> str:
        return value.replace('"', '\\"')

