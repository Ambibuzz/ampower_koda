"""Bounded continuation earned by comparable, executed behavioral checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


def observe(previous: dict | None, report, receipts: list[dict]) -> dict:
    result = deepcopy(previous or {})
    suites = result.setdefault('suites', {})
    active = []
    eligible = bool(receipts) and not report.environment_failures
    if any(r.name.startswith(('tests:contract:', 'tests:frozen:', 'tests:mocked:', 'tests:configuration',
                              'tests:changed-during-run')) for r in report.failures):
        eligible = False
    for receipt in receipts:
        summary = receipt.get('test_summary')
        if receipt.get('timed_out') or (not summary and not receipt.get('passed')):
            eligible = False
        if not summary:
            continue
        if (not summary['passed'] and not summary['failed']) or (not receipt.get('passed') and not summary['failed']):
            eligible = False
        identity = [receipt['name'], receipt['argv'], summary['tests'], summary['skipped']]
        if receipt.get('portable'):
            identity.append(receipt.get('test_revisions', {}))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        failed = summary['failed']
        entry = suites.setdefault(key, {'baseline_failed': list(failed)})
        entry.update(latest_failed=list(failed), total=summary['total'], passed=summary['passed'])
        active.append(key)
    result.update(eligible=eligible and bool(active), active=active)
    return result


def extend(state: dict, *, allowance: int, max_grants: int) -> dict:
    """Explicit/legacy budgets stay fixed; default budgets can earn two slices."""
    grants = int(state.get('automatic_repair_budget_grants') or 0)
    progress = state.get('verification_progress') or {}
    if (not state.get('automatic_repair_budget_enabled') or grants >= max_grants
            or not progress.get('eligible') or not progress.get('active')):
        return {}
    entries = [progress['suites'][key] for key in progress['active']]
    # Regressions and changed test inventories do not count as improvement.
    comparisons = [(set(e['latest_failed']), set(e['baseline_failed'])) for e in entries]
    improved = all(current <= baseline for current, baseline in comparisons) and any(
        current < baseline for current, baseline in comparisons)
    green = all(not current for current, _ in comparisons) and not progress.get('finish_grant_used')
    if not improved and not green:
        return {}
    updated = deepcopy(progress)
    for key in updated['active']:
        updated['suites'][key]['baseline_failed'] = list(updated['suites'][key]['latest_failed'])
    if green:
        updated['finish_grant_used'] = True
    return {'tool_rounds_limit': state['tool_rounds_limit'] + allowance,
            'automatic_repair_budget_grants': grants + 1, 'verification_progress': updated}
