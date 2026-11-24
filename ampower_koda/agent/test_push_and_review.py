"""Tests for pushing, moving the checkout, ending reviews, and the test runner's summary.

Git runs for real in a temporary repository; the database, GitHub and models are faked.
Run with: bench --site <site> run-tests --module ampower_koda.agent.test_push_and_review
"""

import inspect
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from ampower_koda.agent import api, execution_contract, executor, graph, verification


class PatchedTestCase(unittest.TestCase):
    def patch(self, target, name, **options):
        patcher = mock.patch.object(target, name, **options)
        self.addCleanup(patcher.stop)
        return patcher.start()


class GitTestCase(PatchedTestCase):
    """A repository on ``main`` whose branch ``ai/1`` has one more commit."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.repo = Path(folder.name)
        self.git("init", "-b", "main")
        self.git("config", "user.email", "tests@example.com")
        self.git("config", "user.name", "Tests")
        self.commit("one")
        self.git("checkout", "-b", "ai/1")
        self.commit("two")

        from ampower_koda.agent import git_ops
        for module in (executor, git_ops):
            self.patch(module, "get_repo_root", return_value=str(self.repo))
        self.busy_requests = self.patch(executor, "other_busy_requests", return_value=[])

    def git(self, *arguments):
        done = subprocess.run(["git", *arguments], cwd=self.repo, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    def commit(self, text):
        (self.repo / "file.txt").write_text(text)
        self.git("add", ".")
        self.git("commit", "-m", text)

    def current_branch(self):
        return self.git("rev-parse", "--abbrev-ref", "HEAD")


class TestOtherWork(GitTestCase):
    """The checkout moves to a request's branch only when nothing else is using it."""

    def problem(self, branch="ai/1"):
        return executor.other_work_problem("app", branch, "R1")

    def test_clean_idle_checkout_may_move(self):
        self.git("checkout", "main")
        self.assertEqual(self.problem(), "")

    def test_own_uncommitted_work_is_no_problem(self):
        (self.repo / "file.txt").write_text("three")
        self.assertEqual(self.problem(), "")

    def test_uncommitted_work_on_another_branch_is_named(self):
        self.git("checkout", "main")
        (self.repo / "file.txt").write_text("three")
        self.assertIn("file.txt", self.problem())

    def test_running_request_is_named(self):
        self.git("checkout", "main")
        self.busy_requests.return_value = ["R2"]
        self.assertIn("R2", self.problem())
        self.busy_requests.assert_called_once_with("app", "R1")

    def test_running_request_does_not_matter_when_the_checkout_stays(self):
        self.busy_requests.return_value = ["R2"]
        self.assertEqual(self.problem(), "")


class TestDeployPhase(GitTestCase):
    """A push that could not finish can be approved again."""

    def setUp(self):
        super().setUp()
        self.updates = []
        self.doc = mock.Mock(branch_name="ai/1", pr_url="", pr_number=0, user_message="m", agent_plan="")
        config = {"user": "u", "target_app_name": "app", "base_branch": "main", "branch_prefix": "ai/",
                  "git_user_name": "Tests", "git_user_email": "tests@example.com",
                  "github_repo_url": "https://example.com/o/r", "github_token": "t"}
        self.patch(executor, "frappe").get_doc.return_value = self.doc
        self.patch(executor, "log_agent_error")
        self.patch(executor, "check_active")
        self.patch(executor, "_get_doc_config", return_value=config)
        self.patch(executor, "ignored_regression_tests", return_value=[])
        self.patch(executor, "_update_status", side_effect=self.record_update)
        self.push = self.patch(executor, "push_branch", return_value=(True, ""))
        self.create_pr = self.patch(executor, "create_pull_request", return_value=(True, "", "url", 7))

    def record_update(self, request_name, user, status, message="", **fields):
        self.updates.append((status, message, fields.get("error_log")))

    def deploy(self):
        executor.run_deploy_phase.__wrapped__("R1", True, True)
        return self.updates[-1]

    def test_failed_push_awaits_approval_and_the_retry_pushes_the_kept_commit(self):
        (self.repo / "file.txt").write_text("three")
        self.push.return_value = (False, "bad token")
        status, _, error = self.deploy()
        self.assertEqual(status, "Awaiting Push Approval")
        self.assertIn("bad token", error)
        kept_commit = self.git("rev-parse", "ai/1")

        self.push.return_value = (True, "")
        status, message, error = self.deploy()
        self.assertEqual((status, error), ("Completed", ""))
        self.assertIn("PR #7", message)
        self.assertEqual(self.push.call_count, 2)
        self.assertEqual(self.git("rev-parse", "ai/1"), kept_commit)

    def test_failed_pull_request_awaits_approval(self):
        self.create_pr.return_value = (False, "no access", None, None)
        status, _, error = self.deploy()
        self.assertEqual(status, "Awaiting Push Approval")
        self.assertIn("no access", error)

    def test_nothing_to_push_says_so(self):
        self.git("checkout", "main")
        self.git("branch", "-f", "ai/1", "main")
        status, message, _ = self.deploy()
        self.assertEqual((status, message), ("Awaiting Push Approval", "No changes to push."))
        self.push.assert_not_called()

    def test_running_request_keeps_the_checkout(self):
        self.git("checkout", "main")
        self.busy_requests.return_value = ["R2"]
        status, message, _ = self.deploy()
        self.assertEqual(status, "Awaiting Push Approval")
        self.assertIn("R2", message)
        self.assertEqual(self.current_branch(), "main")
        self.push.assert_not_called()


class TestStartPush(PatchedTestCase):
    """Both push buttons queue the same job behind the same guard."""

    def setUp(self):
        self.doc = mock.Mock(status="Awaiting Push Approval", branch_name="ai/1", target_app_name="app")
        self.doc.name = "R1"
        self.frappe = self.patch(api, "frappe")
        self.frappe.get_doc.return_value = self.doc
        self.frappe.throw.side_effect = ValueError
        self.patch(api, "_", side_effect=str)
        self.patch(api, "_require_request_branch")
        self.patch(api, "_has_pushable_changes", return_value=True)
        self.patch(api, "other_busy_requests", return_value=[])
        self.problem = self.patch(api, "other_work_problem", return_value="")
        self.enqueue = self.patch(api, "enqueue_job")

    def push(self, endpoint, **options):
        return inspect.unwrap(endpoint)("R1", **options)

    def test_push_clears_the_last_error_and_queues_the_job(self):
        for endpoint in (api.approve_push, api.ide_push):
            self.frappe.db.set_value.reset_mock()
            self.enqueue.reset_mock()
            self.assertEqual(self.push(endpoint, create_pr="0")["status"], "ok")
            self.frappe.db.set_value.assert_called_once_with(
                api.DOCTYPE_NAME, "R1", {"status": "Pushing", "error_log": ""})
            job = self.enqueue.call_args.kwargs
            self.assertEqual((job["request_name"], job["do_push"], job["do_pr"]), ("R1", True, False))

    def test_other_work_stops_either_button(self):
        self.problem.return_value = "R2 is still running"
        for endpoint in (api.approve_push, api.ide_push):
            with self.assertRaises(ValueError):
                self.push(endpoint)
        self.enqueue.assert_not_called()
        self.frappe.db.set_value.assert_not_called()

    def test_a_dead_request_is_released_before_the_check(self):
        with mock.patch.object(api, "other_busy_requests", return_value=["R2"]), \
                mock.patch.object(api, "_reconcile_if_dead") as reconcile:
            self.push(api.approve_push)
        reconcile.assert_called_once()

    def test_no_action_is_refused(self):
        for endpoint in (api.approve_push, api.ide_push):
            with self.assertRaises(ValueError):
                self.push(endpoint, push_branch=0, create_pr=0)


class TestReviewDecision(unittest.TestCase):
    def decide(self, severity, status="satisfied"):
        payload = {
            "review_passed": False,
            "issues": [{"criterion": 1, "severity": severity, "issue": "Saving breaks the old form."}],
            "evidence": [{"criterion": 1, "status": status, "evidence": "form.js:3"}],
        }
        return execution_contract.review_decision(payload, ["The form saves"])[0]

    def test_blocking_finding_is_repaired_even_when_every_criterion_is_met(self):
        self.assertEqual(self.decide("P0"), "repair")
        self.assertEqual(self.decide("P1"), "repair")

    def test_minor_findings_pass_when_every_criterion_is_met(self):
        self.assertEqual(self.decide("P2"), "pass")

    def test_unmet_criterion_is_repaired(self):
        self.assertEqual(self.decide("P1", status="unmet"), "repair")

    def test_missing_evidence_comes_first(self):
        self.assertEqual(self.decide("P1", status="unverified"), "needs_evidence")


class TestReviewCap(unittest.TestCase):
    """A run that keeps failing review stops with its work kept, before the graph's own step limit."""

    def review(self, earlier_passes):
        state = {"target_app_name": "app", "review_passes": earlier_passes, "turn_exhausted": True,
                 "task_completion": {}, "verification_contract": {}}
        failing = verification.HealthReport([verification.CheckResult("syntax:a.py", False, "line 1")])
        with ExitStack() as patches:
            for name, value in {
                "_execution_context": ({"id": "PLAN", "tasks": []}, ["c"], None),
                "_log_stage": [], "_without_redacted": {}, "change_evidence": ([], ""), "_review_paths": [],
                "_integration_neighbor_paths": [], "renamed_sources": {}, "_approved_deletions": [],
                "run_health_checks": failing, "_publish_agent_log": None, "_persist_task_results": None,
            }.items():
                patches.enter_context(mock.patch.object(graph, name, return_value=value))
            return graph.review_node(state)

    def test_last_allowed_review_stops_the_run(self):
        updates = self.review(graph.MAX_REVIEW_PASSES - 1)
        self.assertIn(f"did not pass {graph.MAX_REVIEW_PASSES} reviews", updates["review_stopped"])
        self.assertEqual(graph.should_retry_implement(updates), "done")

    def test_an_earlier_review_goes_on(self):
        updates = self.review(graph.MAX_REVIEW_PASSES - 2)
        self.assertEqual(updates["review_passes"], graph.MAX_REVIEW_PASSES - 1)
        self.assertNotIn("review_stopped", updates)
        self.assertEqual(graph.should_retry_implement(updates), "implement")

    def test_every_review_fits_in_the_graphs_step_limit(self):
        self.assertLess(1 + 2 * graph.MAX_REVIEW_PASSES, 100)


class TestRunnerSummary(PatchedTestCase):
    """The Python runner's result comes from its summary file, never from what a test prints."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        # As on a bench: <apps>/<app>/<app>
        self.root = Path(folder.name) / "probe_app" / "probe_app"
        self.tests = self.root / ".koda" / "tests"
        self.tests.mkdir(parents=True)
        self.patch(verification, "_app_root", return_value=str(self.root))
        self.patch(verification, "site_environment", return_value={})

    def run_tests(self, body):
        (self.tests / "test_probe.py").write_text(
            "import os, unittest\nclass T(unittest.TestCase):\n    def test_it(self):\n        " + body + "\n")
        health, receipts = verification.run_verification("probe_app", {"config": None, "commands": []})
        return health.passed, receipts

    def test_passing_test_is_counted(self):
        passed, receipts = self.run_tests("self.assertTrue(True)")
        self.assertTrue(passed, receipts)

    def test_failing_test_fails(self):
        passed, _ = self.run_tests("self.fail('no')")
        self.assertFalse(passed)

    def test_printed_summary_does_not_replace_the_file(self):
        forged = json.dumps({"total": 1, "tests": ["x"], "failed": [], "skipped": [], "passed": 1})
        passed, _ = self.run_tests(f"print('KODA_TEST_SUMMARY {forged}', flush=True); os._exit(0)")
        self.assertFalse(passed)

    def test_summary_does_not_depend_on_how_much_is_printed(self):
        passed, receipts = self.run_tests("print('x' * 2_000_000)")
        self.assertTrue(passed, receipts)

    def test_unreadable_summary_is_no_summary(self):
        self.assertIsNone(verification._read_summary(str(self.root / "missing.json")))
        half = self.root / "half.json"
        half.write_text('{"total": 1, "tes')
        self.assertIsNone(verification._read_summary(str(half)))


if __name__ == "__main__":
    unittest.main()
