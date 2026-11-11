"""Optional Playwright verification of new background/snow controls and exports."""

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, default=Path('artifacts/natural-twilight-snow-20261111'))
    parser.add_argument('--library-directory', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if args.library_directory:
        environment['LD_LIBRARY_PATH'] = str(args.library_directory.resolve())
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, env=environment)
        page = browser.new_page(viewport={'width': 1440, 'height': 1080})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until='networkidle')
        page.select_option('#region', 'petersburg-2025')
        page.get_by_text('Снег и отражение поверхности', exact=True).click()
        page.check('#surface-enabled')
        page.fill('#direct-horizontal', '.1')

        def submit(predicate):
            with page.expect_response(lambda r: r.url.endswith('/api/jobs') and r.request.method == 'POST') as pending:
                page.click('#calculate')
            response = pending.value
            assert response.status == 202, response.text()
            job_id = response.json()['id']
            page.wait_for_function('active === null && currentResult !== null && ('+predicate+')', timeout=90000)
            assert page.locator('#message').is_hidden(), page.locator('#message').inner_text()
            return job_id

        # Import pins all inputs and requires readForm/fill to preserve them.
        page.set_input_files('#import', args.output/'bare/scene.json')
        page.wait_for_function('document.querySelector("#surface-enabled").checked')
        first = submit('currentResult.request.surface?.snow_fraction === 0')
        if not page.locator('#snow-fraction').is_visible():
            page.get_by_text('Снег и отражение поверхности', exact=True).click()
        page.fill('#snow-fraction', '1')
        second = submit('currentResult.request.surface?.snow_fraction === 1')
        page.select_option('#compare-left', first)
        page.select_option('#compare-right', second)
        with page.expect_response(lambda r: r.url.endswith('/api/compare')) as pending:
            page.click('#compare')
        comparison = pending.value.json()
        assert pending.value.status == 200 and all(r['right_over_left'] > 1 for r in comparison['directions'])
        page.set_input_files('#import', args.output/'white-night/scene.json')
        page.wait_for_function('document.querySelector("#twilight-mode").value === "patat2006"')
        third = submit('currentResult.request.time === "2026-06-21T22:00:00Z"')
        assert page.evaluate('currentResult.sky.directions.filter(r=>r.supported).length') == 2
        assert 'Сумерки: взгляд ниже 50°' in page.locator('#result').inner_text()
        with page.expect_download() as download:
            page.get_by_text('Сохранить сценарий', exact=True).click()
        saved = args.output/'browser-background-scene.json'
        download.value.save_as(saved)
        exported = json.loads(saved.read_text())
        assert exported['background']['twilight'] == 'patat2006'
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(args.output/'browser-background.png'))
        mobile = browser.new_page(viewport={'width': 390, 'height': 844})
        mobile.on('pageerror', lambda error: errors.append(str(error)))
        mobile.goto(args.url, wait_until='networkidle')
        mobile.wait_for_selector('#result:not([hidden])')
        assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        mobile.screenshot(path=str(args.output/'browser-mobile-background.png'))
        assert not errors, errors
        record = {'browser': 'Chromium / Playwright', 'completed_jobs': 3, 'total_jobs': 3,
            'job_ids': [first, second, third], 'fixed_lamp_snow_comparison': comparison,
            'twilight_partial_support_directions': 2, 'twilight_total_directions': 5,
            'scenario_export_preserves_background': True, 'javascript_errors': errors,
            'mobile_horizontal_overflow': False,
            'desktop_viewport': [1440, 1080], 'mobile_viewport': [390, 844]}
        (args.output/'browser-check.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: v for k, v in record.items() if k not in ['fixed_lamp_snow_comparison', 'job_ids']}))
        browser.close()


if __name__ == '__main__':
    main()
