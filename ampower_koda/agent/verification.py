"""Execute behavioral checks and return evidence to the implementation/review loop.

An app can configure its native integration commands in .koda/verification.json.
Without configuration, portable unittest and Node tests in .koda/tests are run.
Those tests should import the real changed code and replace only external I/O.
Commands are frozen before implementation and are never inferred from model prose.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from .checks import CheckResult, HealthReport
from .run_control import check_active
from .tools import ARCHIVE_DIRECTORY, _app_root, command_environment

CONFIG_PATH = ".koda/verification.json"
GUIDANCE_PATH = ".koda/verification.md"
TEST_DIRECTORY = ".koda/tests"
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 300
MAX_OUTPUT = 8000

# Classify SQL before it reaches the live site: the reason a statement is refused, '' when it
# may run. Comments are dropped (outside quoted text) and every statement is checked, so a
# leading comment or a second statement cannot hide DDL. Refused: statements MariaDB commits
# on its own (DDL, COMMIT, START TRANSACTION, LOCK...), CALL/PREPARE/EXECUTE (their contents
# are unseen), SET of autocommit or transaction state, and executable /*! */ comments.
# ROLLBACK and savepoints stay allowed: they cannot persist anything, and Frappe's own
# rollback() and the runner's teardown use them. read_only (the SQL read probe) also refuses
# writes and SELECT ... INTO OUTFILE/DUMPFILE/@var. This is a filter for MariaDB syntax, not
# a sandbox: raw cursors bypass it, and the real fix is an isolated verification database.
# _commits stays self-contained: the host executes this same source for the read probe.
_SQL_GUARD = r"""
_COMMITTING = {'create', 'alter', 'drop', 'truncate', 'rename', 'commit', 'begin', 'start', 'grant', 'revoke',
               'lock', 'unlock', 'analyze', 'optimize', 'repair', 'flush', 'check', 'cache', 'load', 'reset',
               'purge', 'install', 'uninstall', 'xa', 'call', 'prepare', 'execute', 'deallocate', 'change',
               'stop', 'kill', 'shutdown'}
def _commits(query, read_only=False):
    import re
    text, code, i = str(query or ''), [], 0
    while i < len(text):
        c = text[i]
        if c in '\'"`':
            end = i + 1
            while end < len(text) and text[end] != c:
                end += 2 if text[end] == '\\' and c != '`' else 1
            if end >= len(text):
                return 'an unterminated quoted string'
            code.append(' ? ')  # doubled quotes ('it''s') read as two adjacent literals
            i = end + 1
        elif text.startswith('/*', i):
            if text.startswith(('/*!', '/*M!'), i):
                return 'an executable /*! */ comment'
            end = text.find('*/', i + 2)
            if end < 0:
                return 'an unterminated comment'
            code.append(' ')
            i = end + 2
        elif c == '#' or text.startswith('--', i) and (i + 2 == len(text) or text[i + 2] <= ' '):
            end = text.find('\n', i)
            code.append(' ')
            i = len(text) if end < 0 else end
        else:
            code.append(c)
            i += 1
    statements = [s.strip().lower() for s in ''.join(code).split(';') if s.strip()]
    if read_only and len(statements) != 1:
        return 'more than one statement'
    for statement in statements:
        words = re.findall(r'[a-z_][a-z0-9_]*', statement) + ['', '']
        if words[0] in ('create', 'drop') and words[1] == 'temporary':
            continue  # temporary tables commit nothing
        if words[0] in _COMMITTING:
            return words[0].upper() + ', which commits or escapes the open transaction'
        if words[0] == 'set' and re.search(r'\b(autocommit|transaction|transaction_\w+|tx_\w+|completion_type'
                                           r'|password|statement)\b', statement):
            return 'a SET that changes autocommit or transaction state'
        if read_only:
            if words[0] not in ('select', 'with', 'show', 'describe', 'desc', 'explain'):
                return words[0].upper() + ', which is not a read'
            # a word followed by "(" is a function (REPLACE(), INSERT(), TRUNCATE()), not a statement
            found = re.search(r'\b(into|outfile|dumpfile|insert|update|delete|replace|merge|create|drop|alter'
                              r'|truncate|rename|call|lock|grant|revoke|procedure)\b(?!\s*\()'
                              r'|\b(nextval|setval)\s*\(', statement)
            if found:
                return found.group(0).split('(')[0].strip().upper() + ' inside a read'
    return ''
"""
_SQL_RULES: dict = {}
exec(_SQL_GUARD, _SQL_RULES)

# Connect to the bench site the agent runs on, so tests and calls see the real
# schema and records; commits are disabled and the connection is rolled back.
# SQL that could persist on its own is refused by _SQL_GUARD above, since no
# rollback could undo it; with autocommit off the server opens the next
# transaction itself, so begin() has nothing to do.
#
# Frappe resolves its log files relative to the working directory ("../logs"
# and "<site>/logs"), as bench does, so the runner starts in the sites folder.
# A failure here is the harness's, not the app's: it is reported with a marker
# the host classifies as an environment failure, never as a code defect for
# the model to repair.
_SITE_CONNECT = """
import os, sys, traceback
SITE = os.environ.get('KODA_SITE')
""" + _SQL_GUARD + """
if SITE:
    try:
        os.chdir(os.environ['KODA_SITES_PATH'])
        import frappe
        frappe.init(site=SITE, sites_path=os.environ['KODA_SITES_PATH'])
        frappe.connect()
        frappe.set_user('Administrator')
        frappe.flags.in_test = True
        frappe.in_test = True  # Frappe v16 reads the module attribute; v15 the flag
        frappe.db.commit = lambda *args, **kwargs: None
        frappe.db.begin = lambda *args, **kwargs: None
        _unguarded_sql = frappe.db.sql
        def _guarded_sql(query, *args, **kwargs):
            reason = _commits(query)
            if reason:
                raise frappe.ValidationError(
                    'Koda runner refused "' + ' '.join(str(query).split())[:100] + '": it contains ' + reason
                    + ', which could persist on the live site where no rollback undoes it. Use existing DocTypes '
                    'and fields; schema changes reach the site through bench migrate after the user approves them.')
            return _unguarded_sql(query, *args, **kwargs)
        frappe.db.sql = _guarded_sql
    except Exception:
        traceback.print_exc(limit=6)
        print('KODA_RUNNER_ERROR could not connect to site ' + SITE, flush=True)
        sys.exit(3)
def site_disconnect():
    if SITE:
        frappe.db.rollback()
        frappe.destroy()
"""

_UNITTEST_BODY = """
import importlib.util, json, os, pathlib, sys, unittest
sys.path.insert(0, os.environ['KODA_APP_PARENT'])
if SITE:
    print('KODA_SITE ' + SITE + ': tests run against the live site; database writes are rolled back.', flush=True)
    try:
        contain()  # before the tests import anything, so they bind the held enqueue and sendmail
    except Exception:
        release()
        traceback.print_exc(limit=6)
        print('KODA_RUNNER_ERROR could not contain the tests', flush=True)
        site_disconnect()
        sys.exit(3)
def case_ids(group):
    for case in group:
        if isinstance(case, unittest.TestSuite): yield from case_ids(case)
        else: yield case.id()
try:
    suite = unittest.TestSuite()
    for index, filename in enumerate(sys.argv[1:]):
        sys.path.insert(0, str(pathlib.Path(filename).parent))
        spec = importlib.util.spec_from_file_location('koda_test_' + str(index), filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    tests = sorted(case_ids(suite))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
finally:
    if SITE:
        release()
        print('KODA_CALL_CONTAINED ' + json.dumps(contained()), flush=True)
    site_disconnect()
identity = lambda case: getattr(case, 'test_case', case).id()
failed = sorted({identity(case) for case, _ in result.failures + result.errors} | {identity(case) for case in result.unexpectedSuccesses})
skipped = sorted({identity(case) for case, _ in result.skipped + result.expectedFailures} - set(failed))
summary = {'total': result.testsRun, 'tests': tests, 'failed': failed, 'skipped': skipped,
           'passed': max(0, result.testsRun - len(failed) - len(skipped))}
print('KODA_TEST_SUMMARY ' + json.dumps(summary), flush=True)
sys.exit(0 if result.wasSuccessful() and summary['passed'] else 1)
"""

# What call_method and generated tests do is kept from outliving them where Python can see it:
# writes outside the scratch dir go to an overlay, real deletes/renames are refused, jobs and mail
# are recorded, child processes are refused, and an audit hook blocks bypasses. Sockets stay open
# (the database and Redis use them), so network calls and external services are NOT contained.
_CALL_CONTAINMENT = """
import builtins, errno, importlib, io, pathlib, shutil, smtplib, stat, subprocess
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_real = {name: getattr(os, name) for name in ('open', 'stat', 'lstat', 'mkdir', 'rename', 'replace', 'remove',
         'unlink', 'rmdir', 'listdir', 'scandir', 'access', 'chmod', 'utime', 'truncate') if hasattr(os, name)}
_real_open, _real_rmtree = io.open, shutil.rmtree
_shadowed, _written, _jobs, _mails, _undo = set(), {}, [], [], []
_active, _SCRATCH, _OVERLAY, _DEVNULL = False, None, None, None

def _key(value):
    # absolute, case-folded path; None for a descriptor or a non-path
    if isinstance(value, int):
        return None
    try:
        return os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(value))))
    except (TypeError, ValueError):
        return None

def _outside(key):
    return key is not None and key != _DEVNULL and key != _SCRATCH and not key.startswith(_SCRATCH + os.sep)

def _shadow(key):
    drive, rest = os.path.splitdrive(key)
    return os.path.join(_OVERLAY, drive.replace(':', '').strip(os.sep).replace(os.sep, '_'), rest.lstrip(os.sep))

def _there(path):
    try:
        _real['lstat'](path)
        return True
    except (OSError, ValueError):
        return False

def _is_dir(path):
    try:
        return stat.S_ISDIR(_real['stat'](path).st_mode)
    except (OSError, ValueError):
        return False

def _lives(key):
    # written by this call: the path or a directory above it exists only in the overlay
    while _shadowed and key not in _shadowed:
        parent = os.path.dirname(key)
        if parent == key:
            return False
        key = parent
    return bool(_shadowed)

def _read(value):
    # a path this call wrote is read from its overlay copy
    key = _key(value) if _shadowed else None
    return _shadow(key) if _outside(key) and _lives(key) else value

def _makedirs(path):
    if not _is_dir(path):
        _makedirs(os.path.dirname(path))
        try:
            _real['mkdir'](path)
        except FileExistsError:
            pass

def _write(value, keep=False, exclusive=False):
    # the overlay path that takes a write to a real path; None for the call's own paths
    key = _key(value)
    if not _outside(key):
        return None, key
    target = _shadow(key)
    if not _lives(key):
        name = os.fsdecode(os.fspath(value))
        if exclusive and _there(key):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), name)
        parent = os.path.dirname(key)
        if not (_is_dir(parent) or _is_dir(_shadow(parent))):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name)
        _makedirs(os.path.dirname(target))
        if keep and _there(key):
            with _real_open(key, 'rb') as source, _real_open(target, 'wb') as copy:
                shutil.copyfileobj(source, copy)
    return target, key

def _wrote(key, value):
    _shadowed.add(key)
    _written.setdefault(key, ('directory: ' if _is_dir(_shadow(key)) else 'file: ')
                        + os.path.abspath(os.fsdecode(os.fspath(value))))

def _forget(key):
    _shadowed.difference_update([known for known in _shadowed if known == key or known.startswith(key + os.sep)])

def _open(file, mode='r', *args, **kwargs):
    if _active and isinstance(mode, str) and not isinstance(file, int):
        if any(flag in mode for flag in 'wax+'):
            # append needs no copy of the original; read-write modes do
            target, key = _write(file, '+' in mode and 'w' not in mode, 'x' in mode)
            if target is not None:
                handle = _real_open(target, mode, *args, **kwargs)
                if _there(target) and not _is_dir(target):  # an opener (tempfile's) may open another path
                    _wrote(key, file)
                return handle
        else:
            file = _read(file)
    return _real_open(file, mode, *args, **kwargs)

def _os_open(path, flags, mode=0o777, *, dir_fd=None):
    if _active and dir_fd is None:
        if flags & _WRITE_FLAGS:
            keep = not flags & os.O_TRUNC and bool(flags & os.O_RDWR or not flags & os.O_APPEND)
            target, key = _write(path, keep, bool(flags & os.O_CREAT and flags & os.O_EXCL))
            if target is not None:
                descriptor = _real['open'](target, flags, mode)
                _wrote(key, path)
                return descriptor
        else:
            path = _read(path)
    return _real['open'](path, flags, mode, dir_fd=dir_fd)

def _mkdir(path, mode=0o777, *, dir_fd=None):
    key = _key(path) if _active and dir_fd is None else None
    if not _outside(key):
        return _real['mkdir'](path, mode, dir_fd=dir_fd)
    if _there(_shadow(key) if _lives(key) else key):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), os.fsdecode(os.fspath(path)))
    target, key = _write(path)
    _real['mkdir'](target, mode)
    _wrote(key, path)

def _routed(function):
    def call(*args, **kwargs):
        if _active and args and kwargs.get('dir_fd') is None:
            args = (_read(args[0]),) + args[1:]
        return function(*args, **kwargs)
    return call

def _deleting(function):
    # only what this call wrote can go; a real file reaches the audit hook
    def call(path, *, dir_fd=None):
        key = _key(path) if _active and dir_fd is None else None
        if _outside(key) and _lives(key) and not _there(key):
            function(_shadow(key))
            _forget(key)
            return None
        return function(path, dir_fd=dir_fd)
    return call

def _renaming(function):
    def call(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        key = _key(src) if _active and src_dir_fd is None and dst_dir_fd is None else None
        moved = _outside(key) and _lives(key) and not _there(key)
        if key is None or not moved and _outside(key):
            return function(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        target, written = _write(dst)
        function(_shadow(key) if moved else src, dst if target is None else target)
        if moved:
            _forget(key)
        if target is not None:
            _wrote(written, dst)
    return call

def _rmtree(path, *args, **kwargs):
    key = _key(path) if _active and kwargs.get('dir_fd') is None else None
    if _outside(key) and _lives(key) and not _there(key):
        _real_rmtree(_shadow(key), *args, **kwargs)
        _forget(key)
        return None
    return _real_rmtree(path, *args, **kwargs)
_rmtree.avoids_symlink_attacks = _real_rmtree.avoids_symlink_attacks

def _copying(function):
    # a native file copy (shutil.copy2 on Windows) writes like open()
    def call(src, dst, *args, **kwargs):
        target, key = _write(dst) if _active else (None, None)
        result = function(_read(src) if _active else src, dst if target is None else target, *args, **kwargs)
        if target is not None:
            _wrote(key, dst)
        return result
    return call

# event: (position of a path it changes, position of that path's dir_fd), ...
_CHANGES = {'open': ((0, None),), 'os.mkdir': ((0, 2),), 'os.remove': ((0, 1),), 'os.rmdir': ((0, 1),),
            'shutil.rmtree': ((0, 1),), 'os.rename': ((0, 2), (1, 3)), 'os.link': ((1, 3),),
            'os.symlink': ((1, 2),), 'os.chmod': ((0, 2),), 'os.chown': ((0, 3),), 'os.utime': ((0, 3),),
            'os.truncate': ((0, None),), 'os.chflags': ((0, None),), 'os.setxattr': ((0, None),),
            'os.removexattr': ((0, None),), '_winapi.CopyFile2': ((1, None),)}
_VERBS = {'open': 'write', '_winapi.CopyFile2': 'write', 'os.mkdir': 'create', 'os.link': 'create',
          'os.symlink': 'create', 'os.remove': 'delete', 'os.rmdir': 'delete', 'shutil.rmtree': 'delete',
          'os.rename': 'move'}
_REMOVALS = ('os.remove', 'os.rmdir', 'shutil.rmtree', 'os.rename')
# a child process runs outside every patch here, and whatever it does no rollback undoes
_PROCESS_EVENTS = {'subprocess.Popen', 'os.system', 'os.exec', 'os.spawn', 'os.posix_spawn', 'os.fork',
                   'os.forkpty', 'os.startfile', '_winapi.CreateProcess', 'pty.spawn'}

def _process_refused(name):
    return PermissionError(errno.EPERM, name + ' is blocked in verification: code run by call_method or run_tests '
                           'cannot start child processes, since nothing they do could be rolled back')

def _no_process(name):
    def blocked(*args, **kwargs):
        raise _process_refused(name)
    return blocked

class _BlockedPopen(subprocess.Popen):
    # a class, so code that subclasses or isinstance-checks Popen still imports
    def __init__(self, *args, **kwargs):
        raise _process_refused('subprocess.Popen')

def _at(fd, value):
    # a path relative to a directory descriptor; None where the descriptor cannot be traced
    try:
        return os.path.join(os.readlink('/proc/self/fd/' + str(fd)), os.fsdecode(os.fspath(value)))
    except (OSError, TypeError, ValueError):
        return None

def _guard(event, args):
    # refuse a change outside the scratch dir that did not go through the overlay
    if _active and event in _PROCESS_EVENTS:
        raise _process_refused(event)  # also a Popen or os.system bound before contain()
    changes = _CHANGES.get(event) if _active else None
    if changes is None or event == 'open' and not (args[2] & _WRITE_FLAGS or isinstance(args[1], str)
                                                   and any(flag in args[1] for flag in 'wax+')):
        return
    paths = []
    for index, fd in changes:
        value, fd = args[index], None if fd is None else args[fd]
        if isinstance(value, int):
            continue  # an open descriptor
        if isinstance(fd, int) and fd >= 0:
            value = _at(fd, value)
            if value is None:
                return  # untraceable where there is no /proc
        paths.append(value)
    if event in _REMOVALS and paths and not _there(paths[0]):
        return  # nothing to remove: the call raises FileNotFoundError itself
    for value in paths:
        if _outside(_key(value)):
            raise PermissionError(errno.EACCES, 'call_method runs contained and cannot ' + _VERBS.get(event, 'change')
                                  + ' files outside its scratch directory', os.fsdecode(os.fspath(value)))

def _describe(method):
    if isinstance(method, str):
        return method
    return '.'.join(str(part) for part in (getattr(method, '__module__', None),
                                           getattr(method, '__qualname__', None)) if part) or repr(method)

def _held(original, describe):
    def enqueue(*args, **kwargs):
        if kwargs.get('now') or kwargs.get('is_async') is False:
            return original(*args, **{**kwargs, 'now': True})  # runs here, inside the containment
        _jobs.append(describe(*args, **kwargs))
        return None
    return enqueue

def _job(method=None, *args, **kwargs):
    return _describe(method)

def _document_job(doctype=None, name=None, method=None, *args, **kwargs):
    return '%s %s: %s' % (doctype, name, _describe(method))

def _mail(original):
    def sendmail(*args, **kwargs):
        subject = kwargs.get('subject', args[2] if len(args) > 2 else None)
        _mails.append('%s to %s' % (subject or '(no subject)', kwargs.get('recipients', args[0] if args else None)))
        # queued instead, in the database, which rolls back
        kwargs.update({key: value for key, value in (('now', False), ('delayed', True)) if key in kwargs})
        return original(*args, **kwargs)
    return sendmail

def _smtp_sendmail(self, from_addr, to_addrs, *args, **kwargs):
    _mails.append('SMTP message to %s' % (to_addrs,))
    return {}

def _smtp_send_message(self, msg, from_addr=None, to_addrs=None, *args, **kwargs):
    _mails.append('%s to %s' % (msg.get('Subject') or 'SMTP message', to_addrs or msg.get('To')))
    return {}

def _swap(owner, name, value):
    _undo.append((owner, name, getattr(owner, name)))
    setattr(owner, name, value)

def contain():
    global _active, _SCRATCH, _OVERLAY, _DEVNULL
    sys.dont_write_bytecode = True
    _SCRATCH = os.path.normcase(os.path.realpath(os.environ['KODA_CALL_SCRATCH']))
    _OVERLAY, _DEVNULL = os.path.join(_SCRATCH, 'files'), _key(os.devnull)
    _swap(builtins, 'open', _open)
    _swap(io, 'open', _open)
    _swap(os, 'open', _os_open)
    _swap(os, 'mkdir', _mkdir)
    for name in ('remove', 'unlink', 'rmdir'):
        _swap(os, name, _deleting(_real[name]))
    for name in ('rename', 'replace'):
        _swap(os, name, _renaming(_real[name]))
    for name in ('stat', 'lstat', 'access', 'listdir', 'scandir', 'chmod', 'utime', 'truncate'):
        if name in _real:
            _swap(os, name, _routed(_real[name]))
    for name in ('exists', 'lexists', 'isfile', 'isdir'):
        _swap(os.path, name, _routed(getattr(os.path, name)))
    # Python 3.10's pathlib calls the os functions it bound on import
    accessor = getattr(pathlib, '_NormalAccessor', None)
    for name in ('open', 'stat', 'listdir', 'scandir', 'chmod', 'mkdir', 'unlink', 'rmdir', 'rename', 'replace'):
        if name in vars(accessor or object):
            _swap(accessor, name, staticmethod(io.open if name == 'open' else getattr(os, name)))
    _swap(shutil, 'rmtree', _rmtree)
    windows = sys.modules.get('_winapi')
    if callable(getattr(windows, 'CopyFile2', None)):
        _swap(windows, 'CopyFile2', _copying(windows.CopyFile2))
    _swap(smtplib.SMTP, 'sendmail', _smtp_sendmail)
    _swap(smtplib.SMTP, 'send_message', _smtp_send_message)
    _swap(subprocess, 'Popen', _BlockedPopen)
    for name in ('run', 'call', 'check_call', 'check_output', 'getoutput', 'getstatusoutput'):
        _swap(subprocess, name, _no_process('subprocess.' + name))
    for name in [n for n in dir(os) if n in ('system', 'popen', 'fork', 'forkpty', 'startfile')
                 or n.startswith(('exec', 'spawn', 'posix_spawn'))]:
        if callable(getattr(os, name, None)):
            _swap(os, name, _no_process('os.' + name))
    posix = sys.modules.get('_posixsubprocess')  # multiprocessing starts children with it directly
    if callable(getattr(posix, 'fork_exec', None)):
        _swap(posix, 'fork_exec', _no_process('_posixsubprocess.fork_exec'))
    owners = [frappe]
    try:
        owners.append(importlib.import_module('frappe.utils.background_jobs'))
    except Exception:
        pass  # no job queue to hold
    for owner in owners:
        for name, describe in (('enqueue', _job), ('enqueue_doc', _document_job)):
            if callable(getattr(owner, name, None)):
                _swap(owner, name, _held(getattr(owner, name), describe))
    if callable(getattr(frappe, 'sendmail', None)):
        _swap(frappe, 'sendmail', _mail(frappe.sendmail))
    try:
        queue = importlib.import_module('frappe.email.queue')
    except Exception:
        queue = None
    if callable(getattr(queue, 'flush', None)):
        _swap(queue, 'flush', lambda *args, **kwargs: _mails.append('email queue flush (not run)'))
    sys.addaudithook(_guard)
    _active = True

def release():
    global _active
    _active = False
    while _undo:
        owner, name, value = _undo.pop()
        setattr(owner, name, value)

def contained():
    shown = (list(_written.values())[:5] + ['job: ' + str(job) for job in _jobs[:5]]
             + ['email: ' + str(mail) for mail in _mails[:5]])
    return {'files': len(_written), 'jobs': len(_jobs), 'emails': len(_mails), 'shown': [item[:200] for item in shown]}
"""

_CALL_RUNNER = _SITE_CONNECT + _CALL_CONTAINMENT + """
import json, sys, traceback
sys.path.insert(0, os.environ['KODA_APP_PARENT'])
method, kwargs = sys.argv[1], json.loads(sys.argv[2])
code = 1
try:
    contain()
except Exception:
    release()
    traceback.print_exc(limit=6)
    print('KODA_RUNNER_ERROR could not contain the call', flush=True)
    site_disconnect()
    sys.exit(3)
try:
    # frappe.db is a proxy attribute, not a module get_attr can import.
    target = (getattr(frappe.db, method.rsplit('.', 1)[1]) if method.startswith('frappe.db.')
              else frappe.get_attr(method))
    result = target(**kwargs)
    print('KODA_CALL_RESULT ' + json.dumps(result, default=str, ensure_ascii=False, indent=1), flush=True)
    code = 0
except Exception:
    traceback.print_exc(limit=8)
finally:
    release()  # the rollback and teardown below are the runner's, not the method's
    messages = [m.get('message', m) if isinstance(m, dict) else m for m in (frappe.local.message_log or [])]
    if messages:
        print('KODA_CALL_MESSAGES ' + json.dumps(messages, default=str, ensure_ascii=False), flush=True)
    print('KODA_CALL_CONTAINED ' + json.dumps(contained()), flush=True)
    site_disconnect()
sys.exit(code)
"""
# Tests run under the same containment as call_method, with the same limits (network is not contained).
_UNITTEST_RUNNER = _SITE_CONNECT + _CALL_CONTAINMENT + _UNITTEST_BODY
CALL_TIMEOUT = 60
MAX_CALL_OUTPUT = 6000
# Read-only Frappe calls allowed besides the app's own functions: they let the model see real records
# and columns before designing a query (the same rollback containment applies to them).
READ_PROBES = frozenset({"frappe.get_all", "frappe.get_list", "frappe.db.get_all", "frappe.db.get_value",
                         "frappe.db.get_values", "frappe.db.count", "frappe.db.exists", "frappe.db.sql"})
RUNNER_ERROR = "KODA_RUNNER_ERROR"
CONTAINED_MARKER = "KODA_CALL_CONTAINED "
#: A stopped runner cannot count; its overlay, jobs and emails end with it all the same.
UNCOUNTED = "database writes rolled back; file writes, background jobs and emails discarded"
ENVIRONMENT_NOTE = ("\nThis is a problem in the Koda test environment, not in the app. Do not change the app to "
                    "work around it; report it as a blocker.")


def test_summary(output: str) -> dict | None:
    """Read a checked runner summary; it never overrides a failing exit code."""
    for line in reversed(output.splitlines()):
        if not line.startswith('KODA_TEST_SUMMARY '):
            continue
        try:
            value = json.loads(line[len('KODA_TEST_SUMMARY '):])
            if not isinstance(value, dict): return None
            for key in ('tests', 'failed', 'skipped'):
                items = value.get(key)
                if not isinstance(items, list) or not all(isinstance(item, str) and item for item in items): return None
                if len(set(items)) != len(items): return None
            total = value.get('total')
            failed, skipped, tests = map(set, (value['failed'], value['skipped'], value['tests']))
            if (type(total) is not int or total <= 0 or total != len(tests)
                    or not (failed | skipped) <= tests or failed & skipped
                    or type(value.get('passed')) is not int
                    or value['passed'] != total - len(failed) - len(skipped)):
                return None
            return {**value, 'tests': sorted(tests), 'failed': sorted(failed), 'skipped': sorted(skipped)}
        except (ValueError, TypeError):
            return None
    return None


def _flat_node_summary(output: str) -> dict | None:
    totals = re.findall(r'^# tests (\d+)\s*$', output, re.M)
    cases = re.findall(r'^(ok|not ok) (\d+) - (.+)$', output, re.M)
    if not totals or int(totals[-1]) != len(cases) or not cases:
        return None  # nested TAP needs a reporter-provided structured summary
    tests, failed, skipped = [], [], []
    for status, number, label in cases:
        pending = bool(re.search(r' # (?:SKIP|TODO)\b', label, re.I))
        name = number + ':' + re.split(r' # (?:SKIP|TODO)\b', label, flags=re.I)[0]
        tests.append(name)
        if pending: skipped.append(name)
        elif status == 'not ok': failed.append(name)
    return {'tests': sorted(tests), 'total': len(tests), 'failed': sorted(failed),
            'skipped': sorted(skipped), 'passed': len(tests) - len(failed) - len(skipped)}


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Verification path escapes the app: {relative}")
    return path


def test_files(root: Path) -> list[Path]:
    directory = _inside(root, TEST_DIRECTORY)
    if not directory.is_dir():
        return []
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and (
        (p.name.startswith("test") and p.suffix == ".py")
        or p.name.endswith((".test.js", ".test.cjs", ".test.mjs"))
    ))
    for path in paths:
        _inside(root, path.relative_to(root).as_posix())
    return paths



def _tracked_tests(directory: Path) -> set[str]:
    """Paths under ``directory`` that git tracks, relative to it. Empty when git cannot say."""
    try:
        done = subprocess.run(["git", "-C", str(directory), "ls-files", "-z"], capture_output=True,
                              timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return set()
    if done.returncode:
        return set()
    return {name for name in done.stdout.decode("utf-8", "replace").split("\0") if name}


def archive_untracked_tests(app_name: str, *, label: str = "") -> list[str]:
    """Move tests an earlier agent run left behind (untracked by git) to ``.koda/archive``.

    ``.koda/`` is git-ignored, so a new run would otherwise freeze an old run's tests as
    pre-existing regression tests. Committed tests stay; the rest are moved, not deleted.
    """
    root = Path(_app_root(app_name)).resolve()
    directory = _inside(root, TEST_DIRECTORY)
    if not directory.is_dir():
        return []
    tracked = _tracked_tests(directory)
    leftovers = sorted(p for p in directory.rglob("*")
                       if p.is_file() and p.relative_to(directory).as_posix() not in tracked
                       and "__pycache__" not in p.relative_to(directory).parts)
    if not leftovers:
        return []
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")[:60]
    destination = _inside(root, f"{ARCHIVE_DIRECTORY}/{time.strftime('%Y%m%d-%H%M%S')}" + (f"-{name}" if name else ""))
    moved = []
    for path in leftovers:
        target = destination / path.relative_to(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)
        moved.append(path.relative_to(root).as_posix())
    return moved


def prepare_contract(app_name: str, *, editable_tests=()) -> dict:
    """Snapshot host configuration and pre-existing regression tests once."""
    root = Path(_app_root(app_name)).resolve()
    path = _inside(root, CONFIG_PATH)
    content = path.read_text(encoding="utf-8") if path.is_file() else None
    config = json.loads(content) if content is not None else {}
    if not isinstance(config, dict) or set(config) - {"commands"}:
        raise ValueError(f"{CONFIG_PATH} must contain a commands array")
    commands = config.get("commands", [])
    if not isinstance(commands, list) or len(commands) > 8:
        raise ValueError("verification commands must be an array of at most eight entries")
    checked = []
    for command in commands:
        if not isinstance(command, dict):
            raise ValueError("Each verification command must be an object")
        argv = command.get("argv")
        timeout = command.get("timeout_seconds", DEFAULT_TIMEOUT)
        cwd = command.get("cwd", ".")
        if (not isinstance(argv, list) or not argv
                or not all(isinstance(arg, str) and arg and "\0" not in arg for arg in argv)):
            raise ValueError("Verification argv must be a nonempty array of strings (no shell)")
        if type(timeout) is not int or not 1 <= timeout <= MAX_TIMEOUT:
            raise ValueError(f"Verification timeout_seconds must be between 1 and {MAX_TIMEOUT}")
        if not isinstance(cwd, str) or not _inside(root, cwd).is_dir():
            raise ValueError("Verification cwd must be an existing directory inside the app")
        checked.append({"name": str(command.get("name") or argv[0]),
                        "argv": argv, "cwd": cwd, "timeout_seconds": timeout})
    approved_edits = set(editable_tests)
    guidance = ""
    note = _inside(root, GUIDANCE_PATH)
    if note.is_file():
        with note.open(encoding="utf-8") as handle:
            guidance = GUIDANCE_PATH + "\n" + handle.read(4000)
    return {"config": content, "commands": checked, "guidance": guidance, "existing_tests": {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in test_files(root) if p.relative_to(root).as_posix() not in approved_edits
    }}


def contract_context(contract: dict) -> str:
    commands = [command['name'] for command in contract.get('commands', [])]
    guidance = contract.get('guidance') or ''
    tests = sorted(contract.get('existing_tests') or {})
    inventory = ''
    if tests:
        shown = tests[:24]
        inventory = ('Existing regression tests (already present; read and reuse before adding coverage):\n'
                     + '\n'.join('- ' + path for path in shown) + '\n'
                     + (f'{len(tests) - len(shown)} more under {TEST_DIRECTORY}.\n' if len(tests) > len(shown) else '')
                     + 'Indexed search may omit this hidden directory; that is not evidence that tests are absent.\n')
    if not commands and not guidance and not tests: return ''
    return ('## EXISTING VERIFICATION CONTRACT\n'
            + ('Configured checks: ' + ', '.join(commands) + '.\n' if commands else '')
            + inventory
            + 'Read this contract before choosing interfaces or replacing existing behavior. Use run_tests; '
              'the complete configured suite must pass before the work is reviewed.\n'
            + guidance)


def needs_tests(paths) -> bool:
    """Whether a change set must ship executed tests: only server Python does.

    Client JavaScript tests would need a stubbed browser and Frappe, and prove little about the page.
    """
    return any(p.endswith(".py") and not p.startswith(".koda/") and not p.endswith("__init__.py")
               for p in paths)


def _is_module(root: Path, dotted: str) -> bool:
    """Whether ``dotted`` names a module of the app; ``root`` may be the package or its parent."""
    parts = dotted.split(".")
    for base in (root.joinpath(*parts[1:]), root.joinpath(*parts)):
        if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
            return True
    return False


def _is_patch(func) -> bool:
    return (isinstance(func, ast.Name) and func.id == "patch") or (
        isinstance(func, ast.Attribute) and func.attr == "patch")


def self_mocks(root: Path, path: Path, package: str) -> list[str]:
    """Where a Python test patches the app module it imports to test.

    Such a test passes whatever the code does. Patching another module (an external service) is allowed.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    aliases, tested = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name.split(".")[0] == package:
                    tested.add(name.name)
                    if name.asname:
                        aliases[name.asname] = name.name
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == package:
            for name in node.names:
                dotted = f"{node.module}.{name.name}"
                if _is_module(root, dotted):
                    tested.add(dotted)
                    aliases[name.asname or name.name] = dotted
                else:
                    tested.add(node.module)

    def patched(target) -> str:
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            return target.value if any(target.value.startswith(m + ".") for m in tested) else ""
        if isinstance(target, ast.Name) and target.id in aliases:
            return aliases[target.id]
        return ""

    found = []
    for node in ast.walk(tree):
        what = ""
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            if _is_patch(func) or (isinstance(func, ast.Name) and func.id == "setattr"):
                what = patched(node.args[0])
            elif isinstance(func, ast.Attribute) and func.attr == "object" and _is_patch(func.value):
                what = patched(node.args[0])
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            what = next((patched(t.value) for t in targets
                         if isinstance(t, ast.Attribute) and patched(t.value)), "")
        if what:
            found.append(f"line {node.lineno}: {what}")
    return found


def freeze_passing(contract: dict, receipts: list[dict], root: Path, mocked=()) -> list[str]:
    """Make agent tests that have passed part of the contract; return the newly frozen paths.

    A test the agent can still rewrite is not evidence, so once green it changes only when a reviewer names it.
    """
    frozen = contract.setdefault("frozen_tests", {})
    added = []
    for receipt in receipts:
        if not receipt.get("passed"):
            continue
        argv = set(receipt.get("argv") or [])
        for relative, digest in (receipt.get("test_revisions") or {}).items():
            if (relative in frozen or relative in contract.get("existing_tests", {}) or relative in mocked
                    or str(_inside(root, relative)) not in argv):
                continue
            frozen[relative] = digest
            added.append(relative)
    return added


def release_frozen(contract: dict, text: str) -> list[str]:
    """Unfreeze the agent tests a review finding names, so the repair may correct them."""
    frozen = contract.get("frozen_tests") or {}
    released = [p for p in frozen if p in text or Path(p).name in text]
    for path in released:
        del frozen[path]
    return released


def _stop_process(proc) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=10)
    proc.wait(timeout=10)


def _output(log) -> str:
    size = log.tell()
    log.seek(0)
    if size <= MAX_OUTPUT:
        data = log.read()
    else:
        head = log.read(2000)
        log.seek(-6000, os.SEEK_END)
        data = head + b"\n... output truncated; failure tail follows ...\n" + log.read()
    return data.decode("utf-8", errors="replace")


def _execute(argv: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int, str, bool]:
    """Bound wall time/output; cancellation also terminates the test process tree."""
    check_active(reserve=timeout + 5)
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=os.name == "posix")
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            while proc.poll() is None:
                check_active(max_age=1)
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                try:
                    proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Also kill children left behind by a completed runner on POSIX.
            _stop_process(proc)
        return proc.returncode, _output(log), timed_out


def _node_executed_tests(output: str, files: list[Path], root: Path) -> bool:
    """Node counts an empty file as a passing test; require a named test case."""
    if not re.search(r"^# pass [1-9][0-9]*\s*$", output, re.M):
        return False
    normalize = lambda value: re.sub(r"[\\/]+", "/", value).removeprefix("./").casefold()
    filenames = {normalize(name) for path in files
                 for name in (str(path), path.relative_to(root).as_posix())}
    for line in output.splitlines():
        match = re.match(r"\s*ok \d+ - (.+)", line)
        if not match or re.search(r" # (?:SKIP|TODO)\b", line, re.I):
            continue
        name = normalize(match.group(1))
        if name not in filenames:
            return True
    return False


def site_environment() -> dict:
    """The site this worker serves, for runners that connect to it; empty outside a site."""
    try:
        import frappe
        local = getattr(frappe, "local", None)
        site, sites_path = getattr(local, "site", None), getattr(local, "sites_path", None)
    except (ImportError, RuntimeError):
        return {}
    if not isinstance(site, str) or not site or not isinstance(sites_path, str) or not sites_path:
        return {}
    return {"KODA_SITE": site, "KODA_SITES_PATH": os.path.abspath(sites_path)}


def _runner_environment(root: Path, env: dict | None) -> dict:
    command_env = dict(env if env is not None else command_environment())
    for key in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "GH_TOKEN", "GITHUB_TOKEN"):
        command_env.pop(key, None)
    command_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root.parent), str(root),
                                                          command_env.get("PYTHONPATH", "")]))
    command_env["PYTHONDONTWRITEBYTECODE"] = "1"
    command_env["KODA_APP_PARENT"] = str(root.parent)
    command_env["KODA_TEST_PYTHON"] = sys.executable
    for key, value in site_environment().items():
        command_env.setdefault(key, value)
    return command_env


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit // 3] + f"\n... {len(text) - limit} chars omitted ...\n" + text[-(limit - limit // 3):]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _containment(output: str) -> tuple[str, str, str]:
    """Strip the runner's discarded-side-effects record: ``(output, header phrase, shown items)``."""
    lines = output.splitlines(keepends=True)
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].startswith(CONTAINED_MARKER):
            continue
        try:
            record = json.loads(lines[index][len(CONTAINED_MARKER):])
            files, jobs, emails = (int(record[key]) for key in ("files", "jobs", "emails"))
            shown = "\n".join(str(item) for item in record.get("shown") or [])
        except (ValueError, TypeError, KeyError, AttributeError):
            break
        del lines[index]
        phrase = (f"database writes rolled back; {_plural(files, 'file write')}, "
                  f"{_plural(jobs, 'background job')} and {_plural(emails, 'email')} discarded")
        return "".join(lines), phrase, shown
    return output, UNCOUNTED, ""


def call_method(app_name: str, method: str, kwargs: dict | None = None, *, env: dict | None = None,
                timeout: int = CALL_TIMEOUT, limit: int = MAX_CALL_OUTPUT) -> str:
    """Run one function of the target app against the live site and return what it did.

    This is the check a fixture cannot make: the real query against real
    records, the real permission and validation paths, the real response shape.
    Runs as Administrator in a separate process; DB writes roll back, file writes go to a
    discarded overlay, real deletes/renames are refused, and jobs and email are only recorded.
    SQL that would commit on its own (DDL, COMMIT, START TRANSACTION, LOCK) and child processes
    are refused. Not contained: raw cursor SQL, Redis, realtime, network, native writes.

    ``limit`` bounds the returned text; a helper reading it for a purpose takes more.
    """
    method = str(method or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+", method) or \
            not (method.startswith(app_name + ".") or method in READ_PROBES):
        return (f"CALL_FAILED: method must be a dotted path inside {app_name}, e.g. "
                f"{app_name}.module.file.function, or one of the read probes {', '.join(sorted(READ_PROBES))}.")
    if kwargs is not None and not isinstance(kwargs, dict):
        return "CALL_FAILED: kwargs must be an object of keyword arguments."
    if method == "frappe.db.sql":
        query = str((kwargs or {}).get("query") or "")
        refused = _SQL_RULES["_commits"](query, read_only=True)
        if refused:
            return ("CALL_FAILED: frappe.db.sql here runs one read-only statement (SELECT, WITH, SHOW, DESCRIBE) "
                    f"without INTO or writes; this query has {refused}.")
        kwargs = {"as_dict": True, **kwargs}
    root = Path(_app_root(app_name)).resolve()
    command_env = _runner_environment(root, env)
    if not command_env.get("KODA_SITE"):
        return "RUNTIME_UNAVAILABLE: no Frappe site is connected to this worker; use run_tests instead."
    argv = [sys.executable, "-c", _CALL_RUNNER, method, json.dumps(kwargs or {}, default=str)]
    try:
        # The overlay and the call's temporary files live here and go with it.
        with tempfile.TemporaryDirectory(prefix="koda-call-", ignore_cleanup_errors=True) as scratch:
            scratch = os.path.realpath(scratch)
            temporary = os.path.join(scratch, "tmp")
            os.mkdir(temporary)
            code, output, timed_out = _execute(argv, Path(command_env["KODA_SITES_PATH"]), {
                **command_env, "KODA_CALL_SCRATCH": scratch,
                "TMPDIR": temporary, "TEMP": temporary, "TMP": temporary}, timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"RUNTIME_UNAVAILABLE: could not start the site runner: {type(exc).__name__}: {exc}"
    if RUNNER_ERROR in output:
        return "RUNTIME_UNAVAILABLE: " + _bounded(output, MAX_CALL_OUTPUT) + ENVIRONMENT_NOTE
    output, contained, discarded = _containment(output)
    discarded = "\n[discarded]\n" + discarded if discarded else ""
    if timed_out:
        return (f"CALL_FAILED: {method} did not return within {timeout}s and was stopped ({contained})\n"
                + _bounded(output, limit) + discarded)
    marker = output.find("KODA_CALL_RESULT ")
    if code == 0 and marker >= 0:
        logged, result = output[:marker].strip(), output[marker + len("KODA_CALL_RESULT "):].strip()
        text = f"CALL_OK: {method} on site {command_env['KODA_SITE']} ({contained})\n" + result + discarded
        if logged:
            text += "\n[printed while running]\n" + logged
        return _bounded(text, limit)
    return (f"CALL_FAILED: {method} raised (exit={code}; {contained})\n" + _bounded(output, limit)
            + discarded)


def run_verification(app_name: str, contract: dict, *, env: dict | None = None,
                     required: bool = True) -> tuple[HealthReport, list[dict]]:
    """Nonzero exits are repair findings, never exceptions or model-owned verdicts."""
    root = Path(_app_root(app_name)).resolve()
    results, receipts = [], []
    try:
        path = _inside(root, CONFIG_PATH)
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != contract.get("config"):
            return HealthReport([CheckResult("tests:configuration", False,
                "Verification configuration changed during execution. Restore the original configuration; "
                "the test command cannot be weakened to make a repair pass.")]), []
        for relative, digest in contract.get("existing_tests", {}).items():
            path = _inside(root, relative)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                results.append(CheckResult(f"tests:contract:{relative}", False,
                    "A pre-existing regression test changed or was removed. Restore it and fix the implementation."))
        for relative, digest in (contract.get("frozen_tests") or {}).items():
            path = _inside(root, relative)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                results.append(CheckResult(f"tests:frozen:{relative}", False,
                    "This test passed earlier in this run, so it is now part of the contract. Restore it "
                    "exactly and fix the implementation; add a new test for new behavior."))
        if results:
            return HealthReport(results), []
        files = test_files(root)
        test_revisions = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        mocked = {}
        for path in files:
            relative = path.relative_to(root).as_posix()
            if path.suffix == ".py" and relative not in contract.get("existing_tests", {}):
                found = self_mocks(root, path, app_name)
                if found:
                    mocked[relative] = found
                    results.append(CheckResult(f"tests:mocked:{relative}", False,
                        "This test patches the app module it tests (" + "; ".join(found[:5]) + "), so it "
                        "passes whatever that code does. Remove those patches and run the real code against "
                        "the live site: read existing records, or insert the ones a case needs inside the "
                        "test (writes are rolled back). Patch only external services such as HTTP or email."))
        commands = list(contract.get("commands") or [])
        python_files = [p for p in files if p.suffix == ".py"]
        if python_files:
            commands.append({"name": "unittest", "argv": [sys.executable, "-c", _UNITTEST_RUNNER, *map(str, python_files)],
                             "cwd": ".", "timeout_seconds": DEFAULT_TIMEOUT, "portable": True, 'python': True})
        javascript = [p for p in files if p not in python_files]
        if javascript:
            commands.append({"name": "node:test", "argv": ["node", "--test", "--test-reporter=tap", *map(str, javascript)],
                             "cwd": ".", "timeout_seconds": DEFAULT_TIMEOUT, "node": True, "portable": True})
        if not commands:
            detail = ("No executable behavioral tests were found. Add unittest test*.py under .koda/tests that "
                      "import the changed server code and run it against the live site, with the records each "
                      "case needs read or inserted inside the test (writes are rolled back), then call "
                      "run_tests." if required else
                      "No tests found; none are required because no server Python changed.")
            return HealthReport([CheckResult("tests:missing", not required, detail, verified=False)]), []
        command_env = _runner_environment(root, env)
        for command in commands:
            name, argv = command["name"], command["argv"]
            runner_env = dict(command_env)
            if command.get('portable'):
                # Our portable runners import the outer app package from an
                # explicit root. Host-configured commands keep Python's normal
                # script-directory imports; -P would break their own helpers.
                runner_env['PYTHONSAFEPATH'] = '1'
            try:
                if command.get("python"):
                    # The contained tests' overlay and temporary files live here and go with it.
                    with tempfile.TemporaryDirectory(prefix="koda-tests-", ignore_cleanup_errors=True) as scratch:
                        scratch = os.path.realpath(scratch)
                        temporary = os.path.join(scratch, "tmp")
                        os.mkdir(temporary)
                        code, output, timed_out = _execute(argv, _inside(root, command.get("cwd", ".")), {
                            **runner_env, "KODA_CALL_SCRATCH": scratch,
                            "TMPDIR": temporary, "TEMP": temporary, "TMP": temporary}, command["timeout_seconds"])
                    if runner_env.get("KODA_SITE"):
                        output, contained, discarded = _containment(output)
                        output += f"\n[containment] {contained}" + ("\n" + discarded if discarded else "")
                else:
                    code, output, timed_out = _execute(argv, _inside(root, command.get("cwd", ".")),
                                                     runner_env, command["timeout_seconds"])
                passed = code == 0 and not timed_out
                summary = test_summary(output) or (_flat_node_summary(output) if command.get('node') else None)
                if command.get('python') and (not summary or not summary['passed'] or summary['failed']):
                    passed = False
                    output += '\nThe Python runner must finish and report at least one executed passing test.'
                if summary and summary['failed']:
                    passed = False
                if command.get("node") and not _node_executed_tests(output, javascript, root):
                    passed = False
                    output += "\nNode must execute at least one passing test; zero tests or all skipped is not verification."
                detail = f"exit={code}" + (f"; timeout after {command['timeout_seconds']}s" if timed_out else "")
                detail += "\n" + output
                if RUNNER_ERROR in output:
                    results.append(CheckResult(f"tests:{name}", False, detail + ENVIRONMENT_NOTE, owner="environment"))
                    continue
                results.append(CheckResult(f"tests:{name}", passed, detail))
                receipts.append({"name": name, "argv": argv, "exit_code": code, "passed": passed,
                                 "timed_out": timed_out, "output": output,
                                 "test_revisions": test_revisions, 'test_summary': summary,
                                 'portable': bool(command.get('portable'))})
            except (OSError, subprocess.SubprocessError) as exc:
                results.append(CheckResult(f"tests:{name}", False,
                    f"Test runner could not start: {type(exc).__name__}: {exc}", owner="environment"))
        after = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in test_files(root)}
        config_after = _inside(root, CONFIG_PATH)
        config_text = config_after.read_text(encoding="utf-8") if config_after.is_file() else None
        if after != test_revisions or config_text != contract.get("config"):
            results.append(CheckResult("tests:changed-during-run", False,
                "Test sources or configuration changed while checks ran. Restore the test contract and rerun; "
                "a test process cannot rewrite its own checks to pass."))
        else:
            freeze_passing(contract, receipts, root, mocked)
    except (OSError, ValueError) as exc:
        results.append(CheckResult("tests:environment", False,
                                   f"Cannot prepare behavioral verification: {exc}", owner="environment"))
    return HealthReport(results), receipts
