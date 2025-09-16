PRAGMA foreign_keys=ON;

SELECT load_extension('./vec0.so');

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    parent TEXT,
    dot_path TEXT UNIQUE NOT NULL,
    is_top BOOL NOT NULL,
    added TIMESTAMP DEFAULT current_timestamp
);


CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    arxiv_id TEXT NOT NULL UNIQUE,
    guid TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    pub_date TEXT NOT NULL,
    creator TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS paper_categories (
    paper_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    PRIMARY KEY (paper_id, category_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);


CREATE INDEX IF NOT EXISTS paper_categories_paper_idx ON paper_categories(paper_id);
CREATE INDEX IF NOT EXISTS paper_categories_cat_idx   ON paper_categories(category_id);




-- Fast text search (keywords)
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
USING fts5(
  title,
  description,
  content='papers',
  content_rowid='id',
  tokenize='unicode61'
);

-- semantic vector search
CREATE VIRTUAL TABLE IF NOT EXISTS papers_vec USING vec0(
    paper_id INTEGER PRIMARY KEY,
    embedding FLOAT[768] distance_metric=cosine   
);

-- keep text search tables inline with papers
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
  INSERT INTO papers_fts(rowid, title, description)
  VALUES (new.id, new.title, new.description);
END;

CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
  INSERT INTO papers_fts(papers_fts, rowid, title, description)
  VALUES('delete', old.id, old.title, old.description);

  DELETE FROM papers_vec WHERE papers_vec.paper_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE OF title, description ON papers BEGIN
  INSERT INTO papers_fts(papers_fts, rowid, title, description)
  VALUES('delete', old.id, old.title, old.description);
  INSERT INTO papers_fts(rowid, title, description)
  VALUES (new.id, new.title, new.description);
END;




