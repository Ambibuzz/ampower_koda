"""Request generation fencing, cooperative cancellation, and worker deadlines."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import time
from uuid import uuid4

import frappe

DOCTYPE = "Agent Request"
MODEL_TIMEOUT_SECONDS = 90
MODEL_MAX_RETRIES = 1
MODEL_TIME_RESERVE = MODEL_TIMEOUT_SECONDS * (MODEL_MAX_RETRIES + 1) + 90
WORKER_TIMEOUT_SECONDS = 1800
_current = ContextVar("koda_active_run", default=None)


class RunStopped(BaseException):
    """Control flow, deliberately not swallowed by tool/provider error handlers."""


class RunDeadline(BaseException):
    """Stop before the hard worker deadline, preserving the current checkpoint."""


@dataclass
class ActiveRun:
    request_name: str
    run_id: str
    deadline: float
    journal: object = None
    checked_at: float = float("-inf")


def current_run():
    return _current.get()


@contextmanager
def run_context(request_name: str, run_id: str, timeout: int = WORKER_TIMEOUT_SECONDS):
    token = _current.set(ActiveRun(request_name, run_id, time.monotonic() + max(1, timeout - 30)))
    try:
        yield _current.get()
    finally:
        _current.reset(token)


def check_active(*, reserve: float = 0, max_age: float = 0) -> None:
    """Stop a cancelled or superseded run, or one too close to its deadline.

    ``max_age`` lets a read-only loop reuse a recent check instead of querying on
    every iteration; anything that writes or calls a model keeps the default 0.
    """
    run = current_run()
    if run is None:
        return
    now = time.monotonic()
    if now - run.checked_at >= max_age:
        # Locking reads see the latest committed row under MariaDB's default
        # repeatable-read isolation. Release immediately, before any slow operation.
        row = frappe.db.get_value(DOCTYPE, run.request_name, ["status", "agent_run_id"],
                                  as_dict=True, for_update=True)
        frappe.db.commit()
        if not row or row.get("status") == "Cancelled" or row.get("agent_run_id") != run.run_id:
            raise RunStopped("Request cancelled or superseded by another run.")
        run.checked_at = now
    if reserve and now + reserve >= run.deadline:
        raise RunDeadline("Execution time slice exhausted. Resume Execution continues from the saved checkpoint.")


def set_request_value(request_name: str, values, value=None, **kwargs):
    """Conditional writes keep cancelled/obsolete workers from relabelling a run."""
    run = current_run()
    filters = request_name
    if run and run.request_name == request_name:
        check_active()
        filters = {"name": request_name, "agent_run_id": run.run_id, "status": ["!=", "Cancelled"]}
    return frappe.db.set_value(DOCTYPE, filters, values, value, **kwargs)


def managed_job(fn):
    @wraps(fn)
    def wrapped(request_name: str, *args, run_id: str = "", run_timeout: int = WORKER_TIMEOUT_SECONDS, **kwargs):
        # Old already-queued jobs cannot safely acquire the identity of a newer run.
        if not run_id:
            raise ValueError("This job predates execution fencing. Restart it from the request form.")
        with run_context(request_name, run_id, run_timeout):
            try:
                check_active()
                return fn(request_name, *args, **kwargs)
            except RunStopped:
                return
            except RunDeadline as exc:
                try:
                    message = str(exc) if current_run().journal else (
                        "Worker time budget reached. Retry this phase; "
                        "execution has not produced a resumable checkpoint.")
                    set_request_value(request_name, {"status": "Failed", "error_log": message})
                    frappe.db.commit()
                    owner = frappe.db.get_value(DOCTYPE, request_name, "owner") or "Administrator"
                    check_active()
                    frappe.publish_realtime("agent_progress", {"request_name": request_name,
                        "status": "Failed", "message": message}, user=owner)
                except RunStopped:
                    pass
                return
    return wrapped


def enqueue_job(method: str, *, request_name: str, queue="default", timeout=WORKER_TIMEOUT_SECONDS,
                clear_checkpoint=False, **kwargs):
    """Persist identity before enqueue, so fast workers and cancellation see it.

    ``clear_checkpoint`` is for jobs after the implementation was accepted: a
    failure there must not offer to resume a checkpoint from before a commit or migration.
    """
    run_id = str(uuid4())
    values = {"agent_run_id": run_id, "rq_job_id": ""}
    if clear_checkpoint:
        values["execution_checkpoint"] = ""
    frappe.db.set_value(DOCTYPE, request_name, values)
    frappe.db.commit()
    try:
        job = frappe.enqueue(method, queue=queue, timeout=timeout, job_id=run_id,
                             request_name=request_name, run_id=run_id, run_timeout=timeout, **kwargs)
    except Exception:
        frappe.db.set_value(DOCTYPE, {"name": request_name, "agent_run_id": run_id,
                                     "status": ["!=", "Cancelled"]},
                            {"status": "Failed", "error_log": "Could not enqueue the agent job. Retry the request."})
        frappe.db.commit()
        raise
    job_id = getattr(job, "id", None)
    if job_id:
        frappe.db.set_value(DOCTYPE, {"name": request_name, "agent_run_id": run_id}, "rq_job_id", str(job_id))
        frappe.db.commit()
        # Cancellation may race between publishing the generation and enqueue.
        status = frappe.db.get_value(DOCTYPE, request_name, "status", for_update=True)
        frappe.db.commit()
        if status == "Cancelled":
            stop_job(str(job_id))
    return job


def stop_job(job_id: str) -> None:
    """Best-effort immediate stop; cooperative fencing remains authoritative."""
    if not job_id:
        return
    try:
        from rq.command import send_stop_job_command
        from rq.exceptions import NoSuchJobError
        from rq.job import Job
        from frappe.utils.background_jobs import get_redis_conn
        connection = get_redis_conn()
        try:
            job = Job.fetch(job_id, connection=connection)
        except NoSuchJobError:
            return
        status = job.get_status(refresh=True)
        if status == "started":
            send_stop_job_command(connection, job_id)
        elif status in {"queued", "scheduled", "deferred"}:
            job.cancel()
    except Exception:
        frappe.log_error(title="Agent cancellation", message=frappe.get_traceback())
