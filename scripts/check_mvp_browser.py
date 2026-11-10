"""Live browser acceptance; requires a running server and optional Playwright.

uv run --with playwright python scripts/check_mvp_browser.py --region demo
Install the browser runtime separately; it is not an application dependency.
"""

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--region', default='demo')
    parser.add_argument('--output', type=Path, default=Path('artifacts/mvp-release-20261111'))
    parser.add_argument('--library-directory', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if args.library_directory:
        environment['LD_LIBRARY_PATH'] = str(args.library_directory.resolve())
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, env=environment)
        page = browser.new_page(viewport={'width': 1440, 'height': 1080}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until='networkidle')
        page.select_option('#region', args.region)
        with page.expect_response(lambda response: response.url.endswith('/api/jobs')
                                  and response.request.method == 'POST') as pending:
            page.click('#calculate')
        first = pending.value.json()['id']
        page.wait_for_function('active === null && currentResult !== null && '
            'currentResult.request.atmosphere.aerosol_optical_depth === 0.1', timeout=90000)
        with page.expect_download() as downloaded:
            page.get_by_text('Сохранить сценарий', exact=True).click()
        scenario = args.output / 'browser-scene.json'
        downloaded.value.save_as(scenario)
        assert json.loads(scenario.read_text())['atmosphere']['aerosol_optical_depth'] == .1
        page.fill('[data-atmo="aerosol_optical_depth"]', '0.2')
        with page.expect_response(lambda response: response.url.endswith('/api/jobs')
                                  and response.request.method == 'POST') as pending:
            page.click('#calculate')
        second = pending.value.json()['id']
        page.wait_for_function('active === null && currentResult !== null && '
            'currentResult.request.atmosphere.aerosol_optical_depth === 0.2', timeout=90000)
        page.select_option('#compare-left', first)
        page.select_option('#compare-right', second)
        with page.expect_response(lambda response: response.url.endswith('/api/compare')) as compared:
            page.click('#compare')
        comparison = compared.value.json()
        assert any(r['right_over_left'] is not None and abs(r['right_over_left'] - 1) > 1e-5
                   for r in comparison['directions'])
        page.wait_for_selector('#comparison tbody tr')
        page.evaluate('window.scrollTo(0, 0)')
        page.screenshot(path=str(args.output / 'mvp-interface.png'))
        page.set_input_files('#import', scenario)
        page.wait_for_function('document.querySelector("[data-atmo=aerosol_optical_depth]").value === "0.1"')
        mobile = browser.new_page(viewport={'width': 390, 'height': 844}, device_scale_factor=1)
        mobile.on('pageerror', lambda error: errors.append(str(error)))
        mobile.goto(args.url, wait_until='networkidle')
        mobile.wait_for_selector('#result:not([hidden])')
        assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        mobile.screenshot(path=str(args.output / 'mvp-mobile.png'))
        mobile.locator('#result').scroll_into_view_if_needed()
        mobile.screenshot(path=str(args.output / 'mvp-mobile-result.png'))
        assert not errors, errors
        record = {'browser': 'Chromium / Playwright', 'region': args.region,
            'first_job': first, 'second_job': second, 'submit_completed': 2,
            'comparison_rendered': True, 'comparison_ratios': comparison['directions'],
            'scenario_downloaded': True, 'scenario_import_restored': True,
            'console_errors': errors, 'mobile_horizontal_overflow': False,
            'desktop_viewport': [1440, 1080], 'mobile_viewport': [390, 844],
            'runtime_dependencies': 'Optional browser runtime; local shared libraries supplied separately if required.'}
        (args.output / 'browser-check.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'browser_jobs_completed': 2, 'comparison_checked': True,
                          'scenario_roundtrip': True, 'console_errors': errors,
                          'mobile_horizontal_overflow': False}, ensure_ascii=False))
        browser.close()


if __name__ == '__main__':
    main()
