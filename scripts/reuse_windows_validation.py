"""Reuse a passed suite only when application, tests and build inputs are identical.

This is for retrying packaging checks after a change to the smoke-test script.
Any application/test/dependency/native-build change requires the full suite.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from finalize_windows_release import validate_test_report

ALLOWED = {
    '.github/workflows/deliver-windows.yml',
    'scripts/verify_portable.py',
    'scripts/reuse_windows_validation.py',
    'scripts/finalize_windows_release.py',
}
META = Path('qa/reused-validation.json')


def command(*args):
    return subprocess.check_output(args, encoding='utf-8').strip()


def check_changes(changed):
    unexpected = set(changed) - ALLOWED
    if unexpected:
        raise RuntimeError('Application/test/build inputs changed; run full Windows tests: ' + ', '.join(sorted(unexpected)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id', type=int)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    repo = os.environ['GITHUB_REPOSITORY']
    sha = command('git', 'rev-parse', 'HEAD')
    run = json.loads(command('gh', 'api', f'repos/{repo}/actions/runs/{args.run_id}'))
    if (run['status'] != 'completed' or run['head_repository']['full_name'] != repo
        or run['head_branch'] != 'main' or run['path'] != '.github/workflows/deliver-windows.yml'
        or not re.fullmatch('[0-9a-f]{40}', run['head_sha'])):
        raise RuntimeError('Invalid source validation run')
    source = run['head_sha']
    subprocess.run(['git', 'merge-base', '--is-ancestor', source, sha], check=True)
    if command('git', 'diff', '--name-only', 'HEAD'):
        raise RuntimeError('Tracked files were changed after checkout; full validation is required')
    changed = command('git', 'diff', '--name-only', source, sha).splitlines()
    check_changes(changed)
    jobs = json.loads(command('gh', 'api', f'repos/{repo}/actions/runs/{args.run_id}/jobs?per_page=100'))
    if not any(s['name'] == 'Run regression suite with actual Whisper and Qwen inference' and s['conclusion'] == 'success'
               for j in jobs['jobs'] for s in j['steps']):
        raise RuntimeError('The original full regression suite did not pass')
    metadata = {'source_run': args.run_id, 'source_commit': source, 'package_commit': sha,
                'validation_url': run['html_url'], 'application_and_tests_identical': True,
                'changed_files': changed}
    if args.report:
        cases = validate_test_report(args.report)
        metadata['tests_passed'] = len(cases)
        metadata['report_sha256'] = hashlib.sha256(args.report.read_bytes()).hexdigest()
        shutil.copy2(args.report, 'qa/test-results-windows.xml')
        print(f'Reused {len(cases)} passed Windows tests; application, tests and build inputs match {source}.')
        META.parent.mkdir(exist_ok=True)
        META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        print('Application, tests, dependencies and native-build inputs match the validated source.')


if __name__ == '__main__':
    main()
