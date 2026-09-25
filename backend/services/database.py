"""SQLite persistence: user profiles + daily meal logs (dietary analytics)."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    sex TEXT NOT NULL,
    height_cm REAL NOT NULL,
    weight_kg REAL NOT NULL,
    conditions TEXT NOT NULL DEFAULT '[]',
    allergies TEXT NOT NULL DEFAULT '[]',
    diet_preference TEXT NOT NULL DEFAULT 'none',
    activity_level TEXT NOT NULL DEFAULT 'moderate',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meal_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    logged_at TEXT NOT NULL,
    food TEXT NOT NULL,
    portion_g REAL NOT NULL,
    kcal REAL NOT NULL,
    protein_g REAL NOT NULL,
    carb_g REAL NOT NULL,
    fat_g REAL NOT NULL,
    sodium_mg REAL NOT NULL,
    sugar_g REAL NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_user(profile: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO users (name, age, sex, height_cm, weight_kg, conditions, allergies,
                                   diet_preference, activity_level, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (profile["name"], profile["age"], profile["sex"], profile["height_cm"], profile["weight_kg"],
             json.dumps(profile.get("conditions", [])), json.dumps(profile.get("allergies", [])),
             profile.get("diet_preference", "none"), profile.get("activity_level", "moderate"),
             dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        return cur.lastrowid


def get_user(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["conditions"] = json.loads(d["conditions"])
    d["allergies"] = json.loads(d["allergies"])
    return d


def log_meal_item(user_id: int, nutrition_row: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO meal_logs (user_id, logged_at, food, portion_g, kcal, protein_g, carb_g,
                                       fat_g, sodium_mg, sugar_g)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, dt.datetime.now(dt.timezone.utc).isoformat(), nutrition_row["food"], nutrition_row["portion_g"],
             nutrition_row["kcal"], nutrition_row["protein_g"], nutrition_row["carb_g"],
             nutrition_row["fat_g"], nutrition_row["sodium_mg"], nutrition_row["sugar_g"]),
        )


def daily_summary(user_id: int, date: str | None = None) -> dict:
    date = date or dt.date.today().isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM meal_logs WHERE user_id = ? AND substr(logged_at, 1, 10) = ?",
            (user_id, date),
        ).fetchall()
    totals = {"kcal": 0.0, "protein_g": 0.0, "carb_g": 0.0, "fat_g": 0.0, "sodium_mg": 0.0, "sugar_g": 0.0}
    for r in rows:
        for k in totals:
            totals[k] += r[k]
    totals = {k: round(v, 1) for k, v in totals.items()}
    return {"date": date, "meals_logged": len(rows), **totals}


def history(user_id: int, days: int = 14) -> list[dict]:
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT substr(logged_at,1,10) AS date, food, portion_g, kcal, protein_g, carb_g,
                      fat_g, sodium_mg, sugar_g
               FROM meal_logs WHERE user_id = ? AND substr(logged_at,1,10) >= ?
               ORDER BY logged_at""",
            (user_id, cutoff),
        ).fetchall()
    return [dict(r) for r in rows]
