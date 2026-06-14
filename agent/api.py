import json
import ssl
import urllib.request
import urllib.parse
import urllib.error

from .config import OLLAMA_URL, BINANCE_URL, BINANCE_FUTURES_URL

# macOS Python often lacks the CA bundle needed for Binance's CDN cert chain;
# fall back to an unverified context only when SSL verification fails.
_NOVERIFY_CTX = ssl._create_unverified_context()


def _http_get(url, timeout=15):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            with urllib.request.urlopen(url, timeout=timeout, context=_NOVERIFY_CTX) as r:
                return json.loads(r.read())
        raise


def _binance_get(path, params=None):
    url = f"{BINANCE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return _http_get(url)


def _futures_get(path, params=None):
    url = f"{BINANCE_FUTURES_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return _http_get(url)


def _ollama_post(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())
