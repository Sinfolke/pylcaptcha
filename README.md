# pylcaptcha
`pylcaptcha` is a modern, high-performance, and stealthy Python library designed to bypass modern anti-bot protections (such as Cloudflare Turnstile and Google Recaptcha V2/V3) while seamlessly integrating with high-speed HTTP clients for scraping and API automation.

By combining browser-based kinematic solvers (`Playwright`/`Camoufox`) with asynchronous HTTP clients (`curl-cffi`), `pycaptcha` allows you to solve interactive verification challenges natively and reuse those sessions for lightweight API requests.

---

## 🚀 Features

- **Double-Stage Challenge Solver:** Automatically detects and resolves Cloudflare interstitial gateways and embedded form CAPTCHA widgets.
- **Human-like Kinematics:** Uses physics-driven, organic mouse movements (Bézier curves via `ShyMouse`) to interact with verification challenges, defeating behavioral analysis.
- **Session & State Synchronization:** Automatically extracts cookies, user-agents, and dynamically generated CSRF tokens to keep lightweight HTTP sessions authenticated.
- **Asynchronous Architecture:** Built from the ground up on `asyncio` for maximum throughput.
- **Universal & Extensible:** Provides generic token extraction hooks that allow integration with any web framework (e.g., Laravel, Django).

---
## 📦 Installation

You can install `pycaptcha` directly from PyPI via `pip`:

```bash
pip install pylcaptcha
```
(Ensure you have your system's compatible dependencies or virtual environments set up).

## Quick Start Guide

There are two modules exported:
```python
from pylcaptcha.captcha import Captcha
from pylcaptcha.http import BrowserHTTP
```
The `Captcha` class is raw captcha solver. To use you pass into the Camoufox page and call proper method

The `BrowserHTTP` class is abstraction over `Captcha` specifically designed for Cloudflare captchas. It allows you to bypass it once with re-using of existing token
```python
import asyncio
from pylcaptcha.http import BrowserHTTP
async def main():
    # Initialize the engine utilizing an asynchronous context manager.
    # This guarantees that the browser and connection pools are safely turn down.
    async with BrowserHTTP() as protocol:
        
        # Step 1: Open the landing page, solve the Turnstile challenge, and sync cookies.
        # this will auto solve both turnstile on page load and embedded into page captcha
        print("Solving captcha on loading...")
        await protocol.get('https://my_cf_page', browser=True) 
        
        # Step 2: Surgically extract the CSRF token from the DOM 
        # and attach it securely to outgoing request headers.
        # NOTE: this is different between backends
        await protocol.sync_csrf_token(
            selector='meta[name="csrf-token"]',
            header_name='x-csrf-token'
        )
        
        # Step 3: Fire an authenticated query payload. 
        # Attach the single-use resolution token obtained from the browser step.
        result = await protocol.post('https://my_cf_page/api/v1/check', data={
            'query': 'human_only_query',
            'token': await protocol.get_cf_token()
        })
        
        print("Response Data:")
        print(result.text)

if __name__ == '__main__':
    asyncio.run(main())
```
## How It Works

Passive vs Active Challenges: The library differentiates between Stage 1 (gateway interstitial blocks) and Stage 2 (embedded in-page widgets like the IMEI form check), solving them sequentially when necessary.

Token Lifecycles: - Cloudflare clearance (cf_clearance) cookies are persistent and can be reused organically for session longevity.

CAPTCHA tokens (like Turnstile challenge signatures) are strictly single-use and generated just-in-time via browser DOM interaction.

CSRF Extraction: Generic evaluators locate CSRF meta tags or hidden inputs, mapping them perfectly to headers or cookies according to target framework requirements.

## Contributing

Contributions are welcome! If you find bugs, encounter new anti-bot defense mechanisms, or want to add framework integrations, please open an issue or submit a pull request.
Note that gcaptcha solver is not yet production stable and i need help with AI development and solving improvements