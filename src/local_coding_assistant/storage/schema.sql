-- Execution Store Schema for Local Coding Assistant
-- SQLite DDL with new identity model (Session -> Turn -> Frame)

-- 1. Tools (GLOBAL) - versioned by content_hash
CREATE TABLE IF NOT EXISTS tools (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    schema_json     TEXT NOT NULL,
    risk_class      TEXT NOT NULL,
    description     TEXT,
    version         TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,
    created_at      INTEGER NOT NULL -- UTC epoch milliseconds
);

-- 2. Templates (GLOBAL) - versioned by content_hash
CREATE TABLE IF NOT EXISTS templates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT NOT NULL,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,
    version         TEXT NOT NULL,
    created_at      INTEGER NOT NULL -- UTC epoch milliseconds
);

-- 3. Sessions (new: simplified, no trace_id, execution_mode, etc.)
CREATE TABLE IF NOT EXISTS sessions (
    id                      TEXT PRIMARY KEY,
    user_input              TEXT NOT NULL,
    status                  TEXT NOT NULL,
    started_at              INTEGER NOT NULL, -- UTC epoch milliseconds
    updated_at              INTEGER,
    agent_mode              TEXT NOT NULL,
    model_name              TEXT,
    total_input_tokens      INTEGER DEFAULT 0,
    total_output_tokens     INTEGER DEFAULT 0,
    total_turns             INTEGER DEFAULT 0,
    total_frames            INTEGER DEFAULT 0,
    metadata_json           TEXT,
    context_profile_json    TEXT
);

-- 4. Turns (explicit turn identity)
CREATE TABLE IF NOT EXISTS turns (
    id                      TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    trace_id                TEXT NOT NULL,
    turn_index              INTEGER NOT NULL,
    user_input              TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'running',
    started_at              INTEGER NOT NULL, -- UTC epoch milliseconds
    finished_at             INTEGER,
    paused_at               INTEGER,          -- UTC epoch milliseconds when paused
    pause_reason            TEXT,             -- 'approval' | 'interrupt' | 'deadline' | 'denied'
    resumed_at              INTEGER,          -- UTC epoch milliseconds when resumed
    context_snapshot_id     TEXT,
    config_snapshot_json    TEXT,
    limits_config_json      TEXT,
    total_input_tokens      INTEGER DEFAULT 0,
    total_output_tokens     INTEGER DEFAULT 0,
    total_frames            INTEGER DEFAULT 0,
    message_boundary_json   TEXT,
    final_answer            TEXT,
    UNIQUE (session_id, turn_index)
);

-- 5. Context Snapshots (immutable prompt assembly state)
CREATE TABLE IF NOT EXISTS context_snapshots (
    id                      TEXT PRIMARY KEY,
    turn_id                 TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    schema_version          INTEGER NOT NULL DEFAULT 1,
    tool_set_ref            TEXT,
    template_set_ref        TEXT,
    repo_snapshot_ref       TEXT,
    memory_snapshot_ref     TEXT,
    message_boundary_json   TEXT,
    content_hash            TEXT NOT NULL,
    payload_json            TEXT,
    created_at              INTEGER NOT NULL, -- UTC epoch milliseconds
    UNIQUE (turn_id)
);

-- 6. Session-Tool association
CREATE TABLE IF NOT EXISTS session_tools (
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    tool_id         INTEGER NOT NULL REFERENCES tools(id),
    PRIMARY KEY (session_id, tool_id)
);

-- 7. Session-Template association
CREATE TABLE IF NOT EXISTS session_templates (
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    template_id     INTEGER NOT NULL REFERENCES templates(id),
    PRIMARY KEY (session_id, template_id)
);

-- 8. Messages (transcript: a pure projection of the runtime event stream)
-- dedupe_key gives every message a stable identity derived from its source
-- event (user:{turn_id}, assistant:{frame_id}, tool:{turn_id}:{call_id}) so
-- re-projecting an event is a no-op (INSERT OR IGNORE).
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id         TEXT NOT NULL,
    frame_id        TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tool_call_id    TEXT,
    tool_name       TEXT,
    metadata_json   TEXT,
    dedupe_key      TEXT,
    created_at      INTEGER NOT NULL, -- UTC epoch milliseconds
    UNIQUE (session_id, dedupe_key)
);

-- 9. Frames (new: added turn_id, context_snapshot_id)
CREATE TABLE IF NOT EXISTS frames (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id             TEXT NOT NULL,
    trace_id            TEXT NOT NULL,
    iteration           INTEGER NOT NULL,
    status              TEXT NOT NULL,
    context_snapshot_id TEXT,
    started_at          INTEGER NOT NULL, -- UTC epoch milliseconds
    finished_at         INTEGER,
    duration_ms         INTEGER,
    input_tokens        INTEGER DEFAULT 0,
    output_tokens       INTEGER DEFAULT 0,
    model_name          TEXT,
    finish_reason       TEXT,
    error_message       TEXT,
    pending_call_id     TEXT,
    history_through_seq  INTEGER NOT NULL DEFAULT 0,
    prompt_hash          TEXT,
    UNIQUE (turn_id, iteration)
);

-- 10. Events (new: added turn_id)
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id         TEXT NOT NULL,
    frame_id        TEXT,
    seq             INTEGER NOT NULL,
    type            TEXT NOT NULL,
    data_json       TEXT NOT NULL,
    timestamp       INTEGER NOT NULL -- UTC epoch milliseconds
);

-- 11. Repository Snapshots (persisted repo map & project info for reconstruction)
CREATE TABLE IF NOT EXISTS repo_snapshots (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    target_project_root     TEXT NOT NULL,
    repo_map_json           TEXT NOT NULL,
    project_info_json       TEXT,
    repo_snapshot_hash      TEXT NOT NULL UNIQUE,
    created_at              INTEGER NOT NULL -- UTC epoch milliseconds
);

-- 12. Workspaces (persisted workspace identity)
-- One row per physical workspace root. workspace_id is a stored UUID (not derived
-- from the root spelling) so session resume and baseline lookups survive
-- process restarts, path renames, symlinks and config drift.
CREATE TABLE IF NOT EXISTS workspaces (
    id              TEXT PRIMARY KEY,
    workspace_root  TEXT NOT NULL UNIQUE,
    config_hash     TEXT NOT NULL,
    updated_at      INTEGER NOT NULL -- UTC epoch milliseconds
);

-- 13. Workspace File Baselines (for conflict detection across session resume)
-- Stores baseline file observations (hash, size, mtime, inode) from READ/SNAPSHOT/AGENT_CREATE sources.
-- These are used to detect external modifications when resuming a session.
CREATE TABLE IF NOT EXISTS workspace_file_baselines (
    workspace_id    TEXT NOT NULL,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    file_path       TEXT NOT NULL,
    hash            TEXT NOT NULL,
    size            INTEGER NOT NULL,
    mtime           TEXT NOT NULL, -- ISO 8601 datetime
    inode           TEXT,
    source          TEXT NOT NULL, -- 'read', 'snapshot', 'agent_create'
    observed_at     TEXT NOT NULL, -- ISO 8601 datetime
    turn_id         TEXT,
    frame_id        TEXT,
    PRIMARY KEY (workspace_id, session_id, file_path)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_workspaces_root ON workspaces (workspace_root);
CREATE INDEX IF NOT EXISTS idx_repo_snapshots_hash ON repo_snapshots (repo_snapshot_hash);
CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages (session_id, seq);
CREATE INDEX IF NOT EXISTS idx_messages_turn_seq ON messages (turn_id, seq);
CREATE INDEX IF NOT EXISTS idx_messages_frame ON messages (frame_id);
CREATE INDEX IF NOT EXISTS idx_turns_session_idx ON turns (session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_turns_trace ON turns (trace_id);
CREATE INDEX IF NOT EXISTS idx_turns_context_snapshot ON turns (context_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_context_snapshot_turn ON context_snapshots (turn_id);
CREATE INDEX IF NOT EXISTS idx_context_snapshot_hash ON context_snapshots (content_hash);
CREATE INDEX IF NOT EXISTS idx_frames_session_iter ON frames (session_id, iteration);
CREATE INDEX IF NOT EXISTS idx_frames_turn_iter ON frames (turn_id, iteration);
CREATE INDEX IF NOT EXISTS idx_frames_trace ON frames (trace_id);
CREATE INDEX IF NOT EXISTS idx_frames_context_snapshot ON frames (context_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_events_session_timestamp ON events (session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events (session_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_turn_timestamp ON events (turn_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_turn_seq ON events (turn_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_frame ON events (frame_id);
CREATE INDEX IF NOT EXISTS idx_events_trace ON events (trace_id);
CREATE INDEX IF NOT EXISTS idx_events_type_timestamp ON events (type, timestamp);
CREATE INDEX IF NOT EXISTS idx_workspace_baselines_workspace ON workspace_file_baselines (workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_baselines_session ON workspace_file_baselines (session_id);
CREATE INDEX IF NOT EXISTS idx_workspace_baselines_turn ON workspace_file_baselines (turn_id);