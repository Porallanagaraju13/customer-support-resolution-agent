"""Optional: drive the UI with headless Chromium and save screenshots (pip install playwright)."""
import sys
from playwright.sync_api import sync_playwright

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    pg.goto(base)
    pg.click("text=large refund")
    pg.click("#send")
    pg.wait_for_selector(".approval-card", timeout=20000)
    pg.screenshot(path="docs/ui_pending_approval.png", full_page=True)
    pg.fill(".approval-card .note", "Verified defect")
    pg.click(".approval-card button.ap")
    pg.wait_for_selector("text=Case closed: refunded", timeout=20000)
    pg.wait_for_timeout(1500)
    pg.screenshot(path="docs/ui_resolved.png", full_page=True)
    b.close()
print("ok")
