from sqlalchemy import text
from app import app, db  # adjust import if your module name differs

with app.app_context():
    db.session.execute(text("ALTER TABLE datasets ADD COLUMN search_query TEXT"))
    db.session.commit()
