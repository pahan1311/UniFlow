"""
db_utils.py
-----------
Simple SQLite storage for submitted inquiries + human review status.
This is the "feedback loop" table your execution plan (Phase 5.2.4)
describes: every inquiry submitted through the dashboard is stored
here, and reviewers can Accept or Correct the model's prediction.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "feedback.db"
DB_PATH.parent.mkdir(exist_ok=True, parents=True)


def init_db():
    """Creates the inquiries table if it doesn't already exist.
    Safe to call every time the app starts.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inquiries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text TEXT,
            days_to_deadline INTEGER,
            predicted_category TEXT,
            predicted_urgency TEXT,
            category_confidence REAL,
            urgency_confidence REAL,
            trigger_keywords TEXT,
            human_review_status TEXT DEFAULT 'pending',
            corrected_category TEXT,
            corrected_urgency TEXT,
            created_at TEXT
        )
        """
    )
    # Lightweight migrations keep existing demo databases compatible.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(inquiries)")}
    migrations = {
        "department": "TEXT",
        "queue_status": "TEXT DEFAULT 'New'",
        "assignee": "TEXT DEFAULT 'Unassigned'",
        "updated_at": "TEXT",
    }
    for column, definition in migrations.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE inquiries ADD COLUMN {column} {definition}")
    conn.execute(
        "UPDATE inquiries SET department = COALESCE(corrected_category, predicted_category) "
        "WHERE department IS NULL OR department = ''"
    )
    conn.commit()
    conn.close()


def insert_inquiry(raw_text, predicted_category, predicted_urgency,
                    category_confidence, urgency_confidence, trigger_keywords,
                    department=None):
    """Saves a new submitted inquiry with the model's predictions.
    Returns the new row's id, so the UI can later update it via
    Accept/Correct buttons.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO inquiries
            (raw_text, days_to_deadline, predicted_category, predicted_urgency,
             category_confidence, urgency_confidence, trigger_keywords, department,
             queue_status, assignee, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'New', 'Unassigned', ?, ?)
        """,
        (
            raw_text, None, predicted_category, predicted_urgency,
            category_confidence, urgency_confidence, ";".join(trigger_keywords),
            department or predicted_category,
            datetime.now().isoformat(timespec="seconds"),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def update_review(inquiry_id: int, status: str, corrected_category=None, corrected_urgency=None):
    """Called when a human clicks Accept or Correct in the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        UPDATE inquiries
        SET human_review_status = ?, corrected_category = ?, corrected_urgency = ?
        WHERE id = ?
        """,
        (status, corrected_category, corrected_urgency, inquiry_id),
    )
    conn.commit()
    conn.close()


def update_queue_item(inquiry_id: int, queue_status: str, assignee: str):
    """Updates workflow fields from a department dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE inquiries SET queue_status = ?, assignee = ?, updated_at = ? WHERE id = ?",
        (queue_status, assignee, datetime.now().isoformat(timespec="seconds"), inquiry_id),
    )
    conn.commit()
    conn.close()


def fetch_all() -> pd.DataFrame:
    """Returns every stored inquiry as a DataFrame, newest first."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM inquiries ORDER BY id DESC", conn)
    conn.close()
    return df
