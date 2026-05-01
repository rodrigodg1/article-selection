"""
Vercel serverless entry: WSGI app + DB init on cold start.

SQLite on Vercel must use a writable path (see app.py when VERCEL=1).
"""
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app import app as flask_app  # noqa: E402
from app import db  # noqa: E402

with flask_app.app_context():
    db.create_all()

app = flask_app
