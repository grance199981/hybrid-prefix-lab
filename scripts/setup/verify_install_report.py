"""Check pip install reports against the official PyPI JSON API.

For every archive pip downloaded (from any index or mirror), compare its sha256
with the digests pypi.org publishes for that project and version. Packages that
are not on PyPI at that version (e.g. the vLLM per-commit wheel) are listed
separately. Standard library only; run where pypi.org is reachable.

Usage: python scripts/setup/verify_install_report.py report.json [report2.json ...]
Exit code 1 when any archive does not match.
"""
import json
import sys
import time
import urllib.error
import urllib.request

RETRIES = 6


def pypi_digests(name, version):
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    req = urllib.request.Request(url, headers={"User-Agent": "install-report-check"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return {u["digests"]["sha256"] for u in json.load(r)["urls"]}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == RETRIES - 1:
                raise
        except (urllib.error.URLError, OSError):
            # Slow or flaky route to pypi.org: retry, never treat as a match.
            if attempt == RETRIES - 1:
                raise
        time.sleep(2 * (attempt + 1))


def main():
    ok, bad, not_on_pypi, no_hash = [], [], [], []
    for path in sys.argv[1:]:
        for item in json.load(open(path, encoding="utf-8"))["install"]:
            name, version = item["metadata"]["name"], item["metadata"]["version"]
            info = item.get("download_info", {})
            sha = info.get("archive_info", {}).get("hashes", {}).get("sha256")
            if not sha:
                no_hash.append((name, version, info.get("url")))
                continue
            digests = pypi_digests(name, version)
            if digests is None:
                not_on_pypi.append((name, version, info.get("url")))
            elif sha in digests:
                ok.append((name, version))
            else:
                bad.append((name, version, info.get("url"), sha))
    print(f"matched pypi.org: {len(ok)}")
    print(f"not on pypi.org at that version: {len(not_on_pypi)}")
    for x in not_on_pypi:
        print("   ", *x)
    print(f"no sha256 in report: {len(no_hash)}")
    for x in no_hash:
        print("   ", *x)
    print(f"MISMATCH: {len(bad)}")
    for x in bad:
        print("   ", *x)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
