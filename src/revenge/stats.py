import sqlite3
from pathlib import Path


class Stats:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS matches ("
                "id INTEGER PRIMARY KEY, finished TEXT DEFAULT CURRENT_TIMESTAMP, "
                "brawler TEXT NOT NULL, result TEXT NOT NULL "
                "CHECK(result IN ('victory','defeat','draw')))"
            )
            self.db.commit()
        except sqlite3.Error:
            self.db.close()
            raise

    def record(self, brawler, result):
        with self.db:
            self.db.execute("INSERT INTO matches(brawler,result) VALUES (?,?)", (brawler, result))

    def summary(self):
        counts = dict(self.db.execute("SELECT result,COUNT(*) FROM matches GROUP BY result"))
        return {key: counts.get(key, 0) for key in ("victory", "defeat", "draw")}

    def close(self):
        self.db.close()
