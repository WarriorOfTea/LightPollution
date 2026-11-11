"""Exercise spectral UI, cross-point comparison and a saved ground calibration."""

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, default=Path('artifacts/photometry-spectra-20261111'))
    parser.add_argument('--library-directory', type=Path)
    args = parser.parse_args()
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
        page.get_by_text('Спектральный расчёт', exact=True).click()
        page.select_option('#spectrum-mode', 'on')

        def submit(predicate):
            with page.expect_response(lambda r: r.url.endswith('/api/jobs') and r.request.method == 'POST') as pending:
                page.click('#calculate')
            response = pending.value
            assert response.status == 202, response.text()
            job_id = response.json()['id']
            page.wait_for_function('active === null && currentResult !== null && ('+predicate+')', timeout=90000)
            assert page.locator('#message').is_hidden(), page.locator('#message').inner_text()
            return job_id

        first = submit('currentResult.spectral && currentResult.request.observer.latitude_deg === 60.3')
        page.fill('#latitude', '60.65')
        page.fill('#spectral-mix', '{"HP1":1}')
        second = submit('currentResult.spectral && currentResult.request.observer.latitude_deg === 60.65')
        page.select_option('#compare-left', first)
        page.select_option('#compare-right', second)
        with page.expect_response(lambda r: r.url.endswith('/api/compare')) as pending:
            page.click('#compare')
        comparison = pending.value.json()
        assert pending.value.status == 200 and comparison['left_observer'] != comparison['right_observer']
        page.wait_for_selector('#comparison tbody tr')
        with page.expect_download() as download:
            page.get_by_text('Сохранить сценарий', exact=True).click()
        path = args.output/'browser-spectrum-scene.json'
        download.value.save_as(path)
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(args.output/'browser-spectrum.png'))
        page.set_input_files('#import', path)
        page.wait_for_function('document.querySelector("#spectral-mix").value === \'{"HP1":1}\'')
        mobile = browser.new_page(viewport={'width': 390, 'height': 844})
        mobile.on('pageerror', lambda error: errors.append(str(error)))
        mobile.goto(args.url, wait_until='networkidle')
        mobile.wait_for_selector('#result:not([hidden])')
        assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        mobile.get_by_text('Спектральный отклик города', exact=True).scroll_into_view_if_needed()
        mobile.screenshot(path=str(args.output/'browser-mobile-spectrum.png'))

        # Importing a fitted scene must preserve its source coefficient.
        page.set_input_files('#import', args.output/'madrid-calibrated-validation/scene.json')
        page.wait_for_function('photometricCalibration !== null')
        third = submit('currentResult.calibration.origin === "provided_calibration"')
        assert page.locator('#result').inner_text().find('Подключена калибровка') >= 0
        assert page.locator('#anchor').is_disabled()
        assert not errors, errors
        record = {'browser': 'Chromium / Playwright', 'completed_jobs': 3, 'total_jobs': 3,
            'job_ids': [first, second, third], 'cross_observer_spectral_comparison': comparison,
            'spectral_scene_roundtrip': True, 'ground_calibration_import_applied': True,
            'mobile_horizontal_overflow': False, 'javascript_errors': errors,
            'desktop_viewport': [1440, 1080], 'mobile_viewport': [390, 844]}
        (args.output/'browser-check.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: v for k, v in record.items() if k not in ['cross_observer_spectral_comparison', 'job_ids']}))
        browser.close()


if __name__ == '__main__':
    main()
