//! Bounded, local-only persistent state for the Oreo device runtime.
//!
//! This crate accepts typed operational metadata and explicit, bounded memory
//! records. It has no API for credentials, raw conversation transcripts,
//! model responses, tool payloads, or raw audio.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, params};

const SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateLimits {
    pub max_events: usize,
    pub max_active_timers: usize,
    pub max_timer_history: usize,
    pub max_memories: usize,
    pub max_memory_bytes: usize,
}

impl StateLimits {
    #[must_use]
    pub const fn sbc() -> Self {
        Self {
            max_events: 2_048,
            max_active_timers: 128,
            max_timer_history: 256,
            max_memories: 256,
            max_memory_bytes: 512,
        }
    }

    fn validate(self) -> Result<Self, StateError> {
        if self.max_events == 0
            || self.max_active_timers == 0
            || self.max_timer_history == 0
            || self.max_memories == 0
            || self.max_memory_bytes == 0
        {
            return Err(StateError::new(
                StateErrorKind::InvalidInput,
                "state limits must be positive",
            ));
        }
        Ok(self)
    }
}

pub trait Clock {
    fn now_ms(&self) -> u64;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
}

#[derive(Debug, Default)]
pub struct ManualClock {
    now_ms: AtomicU64,
}

impl ManualClock {
    #[must_use]
    pub const fn new(now_ms: u64) -> Self {
        Self {
            now_ms: AtomicU64::new(now_ms),
        }
    }

    pub fn set(&self, now_ms: u64) {
        self.now_ms.store(now_ms, Ordering::Release);
    }

    pub fn advance(&self, duration: Duration) {
        let increment = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
        let _ = self
            .now_ms
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_add(increment))
            });
    }
}

impl Clock for ManualClock {
    fn now_ms(&self) -> u64 {
        self.now_ms.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeEventKind {
    RuntimeStarted,
    AgentTurn,
    ToolCall,
    TimerScheduled,
    TimerFired,
    Fault,
}

impl RuntimeEventKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::RuntimeStarted => "runtime_started",
            Self::AgentTurn => "agent_turn",
            Self::ToolCall => "tool_call",
            Self::TimerScheduled => "timer_scheduled",
            Self::TimerFired => "timer_fired",
            Self::Fault => "fault",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventOutcome {
    Succeeded,
    Denied,
    Cancelled,
    Failed,
}

impl EventOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Denied => "denied",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimerStatus {
    Scheduled,
    Fired,
    Cancelled,
}

impl TimerStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Scheduled => "scheduled",
            Self::Fired => "fired",
            Self::Cancelled => "cancelled",
        }
    }

    fn parse(value: &str) -> Result<Self, StateError> {
        match value {
            "scheduled" => Ok(Self::Scheduled),
            "fired" => Ok(Self::Fired),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(StateError::new(
                StateErrorKind::Database,
                "stored timer state is invalid",
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimerRecord {
    pub id: String,
    pub due_at_ms: u64,
    pub created_at_ms: u64,
    pub status: TimerStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryScope {
    Session,
    LongTerm,
}

impl MemoryScope {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::LongTerm => "long_term",
        }
    }

    fn parse(value: &str) -> Result<Self, StateError> {
        match value {
            "session" => Ok(Self::Session),
            "long_term" => Ok(Self::LongTerm),
            _ => Err(StateError::new(
                StateErrorKind::Database,
                "stored memory scope is invalid",
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryRecord {
    pub id: String,
    pub scope: MemoryScope,
    pub content: String,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub expires_at_ms: Option<u64>,
}

pub struct StateStore {
    connection: Connection,
    limits: StateLimits,
}

impl StateStore {
    /// Opens or creates a local `SQLite` database and applies known migrations.
    ///
    /// # Errors
    ///
    /// Returns a typed, redacted error for invalid limits, filesystem failure,
    /// unsupported schema versions, or `SQLite` failures.
    pub fn open(path: impl AsRef<Path>, limits: StateLimits) -> Result<Self, StateError> {
        let limits = limits.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|_| {
                StateError::new(
                    StateErrorKind::Database,
                    "state directory could not be created",
                )
            })?;
        }
        let connection = Connection::open(path).map_err(database_error)?;
        Self::from_connection(connection, limits)
    }

    /// Creates an in-memory store for deterministic tests and diagnostics.
    ///
    /// # Errors
    ///
    /// Returns a redacted error if migrations cannot be applied.
    pub fn in_memory(limits: StateLimits) -> Result<Self, StateError> {
        let limits = limits.validate()?;
        let connection = Connection::open_in_memory().map_err(database_error)?;
        Self::from_connection(connection, limits)
    }

    fn from_connection(connection: Connection, limits: StateLimits) -> Result<Self, StateError> {
        connection
            .busy_timeout(Duration::from_secs(2))
            .map_err(database_error)?;
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .map_err(database_error)?;
        let mut store = Self { connection, limits };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&mut self) -> Result<(), StateError> {
        let version = self
            .connection
            .pragma_query_value(None, "user_version", |row| row.get::<_, u32>(0))
            .map_err(database_error)?;
        match version {
            0 => self
                .connection
                .execute_batch(MIGRATION_1)
                .and_then(|()| self.connection.execute_batch(MIGRATION_2))
                .map_err(database_error),
            1 => self
                .connection
                .execute_batch(MIGRATION_2)
                .map_err(database_error),
            SCHEMA_VERSION => Ok(()),
            _ => Err(StateError::new(
                StateErrorKind::UnsupportedSchema,
                "state database schema is newer than this runtime",
            )),
        }
    }

    #[must_use]
    pub const fn schema_version() -> u32 {
        SCHEMA_VERSION
    }

    /// Persists one content-free runtime event and trims the oldest records.
    ///
    /// # Errors
    ///
    /// Returns a redacted database error if the transaction fails.
    pub fn record_event(
        &mut self,
        clock: &dyn Clock,
        kind: RuntimeEventKind,
        outcome: EventOutcome,
    ) -> Result<(), StateError> {
        let at_ms = to_sql_time(clock.now_ms())?;
        let max_events = to_sql_count(self.limits.max_events)?;
        let transaction = self.connection.transaction().map_err(database_error)?;
        transaction
            .execute(
                "INSERT INTO runtime_events (at_ms, kind, outcome) VALUES (?1, ?2, ?3)",
                params![at_ms, kind.as_str(), outcome.as_str()],
            )
            .map_err(database_error)?;
        transaction
            .execute(
                "DELETE FROM runtime_events
                 WHERE sequence NOT IN (
                    SELECT sequence FROM runtime_events ORDER BY sequence DESC LIMIT ?1
                 )",
                [max_events],
            )
            .map_err(database_error)?;
        transaction.commit().map_err(database_error)
    }

    /// Schedules a metadata-only timer that survives process restart.
    ///
    /// # Errors
    ///
    /// Rejects invalid or duplicate identifiers, past deadlines, full timer
    /// capacity, and database failures.
    pub fn schedule_timer(
        &mut self,
        clock: &dyn Clock,
        id: &str,
        due_at_ms: u64,
    ) -> Result<TimerRecord, StateError> {
        validate_timer_id(id)?;
        let now_ms = clock.now_ms();
        if due_at_ms <= now_ms {
            return Err(StateError::new(
                StateErrorKind::InvalidInput,
                "timer deadline must be in the future",
            ));
        }
        let existing = self
            .connection
            .query_row("SELECT 1 FROM timers WHERE id = ?1", [id], |_| Ok(()))
            .optional()
            .map_err(database_error)?;
        if existing.is_some() {
            return Err(StateError::new(
                StateErrorKind::AlreadyExists,
                "timer id is already in use",
            ));
        }
        let active: i64 = self
            .connection
            .query_row(
                "SELECT COUNT(*) FROM timers WHERE status = 'scheduled'",
                [],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        let active = from_sql_count(active)?;
        if active >= self.limits.max_active_timers {
            return Err(StateError::new(
                StateErrorKind::Capacity,
                "active timer limit reached",
            ));
        }
        let record = TimerRecord {
            id: id.to_owned(),
            due_at_ms,
            created_at_ms: now_ms,
            status: TimerStatus::Scheduled,
        };
        self.connection
            .execute(
                "INSERT INTO timers (id, due_at_ms, created_at_ms, status)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    record.id,
                    to_sql_time(record.due_at_ms)?,
                    to_sql_time(record.created_at_ms)?,
                    record.status.as_str()
                ],
            )
            .map_err(database_error)?;
        Ok(record)
    }

    /// Returns scheduled timers due at or before the supplied clock value.
    ///
    /// # Errors
    ///
    /// Returns a redacted database error for query or decoding failures.
    pub fn due_timers(&self, clock: &dyn Clock) -> Result<Vec<TimerRecord>, StateError> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT id, due_at_ms, created_at_ms, status
                 FROM timers
                 WHERE status = 'scheduled' AND due_at_ms <= ?1
                 ORDER BY due_at_ms, id
                 LIMIT ?2",
            )
            .map_err(database_error)?;
        let rows = statement
            .query_map(
                params![
                    to_sql_time(clock.now_ms())?,
                    to_sql_count(self.limits.max_active_timers)?
                ],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .map_err(database_error)?;
        rows.map(|row| {
            let (id, due_at_ms, created_at_ms, status) = row.map_err(database_error)?;
            Ok(TimerRecord {
                id,
                due_at_ms: from_sql_time(due_at_ms)?,
                created_at_ms: from_sql_time(created_at_ms)?,
                status: TimerStatus::parse(&status)?,
            })
        })
        .collect()
    }

    /// Lists all scheduled timers in deterministic deadline order.
    ///
    /// # Errors
    ///
    /// Returns a redacted database error for query or decoding failures.
    pub fn scheduled_timers(&self) -> Result<Vec<TimerRecord>, StateError> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT id, due_at_ms, created_at_ms, status
                 FROM timers
                 WHERE status = 'scheduled'
                 ORDER BY due_at_ms, id
                 LIMIT ?1",
            )
            .map_err(database_error)?;
        let rows = statement
            .query_map([to_sql_count(self.limits.max_active_timers)?], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(database_error)?;
        rows.map(|row| {
            let (id, due_at_ms, created_at_ms, status) = row.map_err(database_error)?;
            Ok(TimerRecord {
                id,
                due_at_ms: from_sql_time(due_at_ms)?,
                created_at_ms: from_sql_time(created_at_ms)?,
                status: TimerStatus::parse(&status)?,
            })
        })
        .collect()
    }

    /// Marks a scheduled timer as fired and bounds terminal timer history.
    ///
    /// # Errors
    ///
    /// Returns not-found when the timer is absent or no longer scheduled.
    pub fn mark_timer_fired(&mut self, id: &str) -> Result<(), StateError> {
        self.finish_timer(id, TimerStatus::Fired)
    }

    /// Cancels a scheduled timer and bounds terminal timer history.
    ///
    /// # Errors
    ///
    /// Returns not-found when the timer is absent or no longer scheduled.
    pub fn cancel_timer(&mut self, id: &str) -> Result<(), StateError> {
        self.finish_timer(id, TimerStatus::Cancelled)
    }

    fn finish_timer(&mut self, id: &str, status: TimerStatus) -> Result<(), StateError> {
        validate_timer_id(id)?;
        let history_limit = to_sql_count(self.limits.max_timer_history)?;
        let transaction = self.connection.transaction().map_err(database_error)?;
        let updated = transaction
            .execute(
                "UPDATE timers SET status = ?1 WHERE id = ?2 AND status = 'scheduled'",
                params![status.as_str(), id],
            )
            .map_err(database_error)?;
        if updated == 0 {
            return Err(StateError::new(
                StateErrorKind::NotFound,
                "scheduled timer was not found",
            ));
        }
        transaction
            .execute(
                "DELETE FROM timers
                 WHERE status != 'scheduled' AND id NOT IN (
                    SELECT id FROM timers
                    WHERE status != 'scheduled'
                    ORDER BY due_at_ms DESC, id DESC
                    LIMIT ?1
                 )",
                [history_limit],
            )
            .map_err(database_error)?;
        transaction.commit().map_err(database_error)
    }

    /// Returns the number of retained runtime events for diagnostics.
    ///
    /// # Errors
    ///
    /// Returns a redacted database error when the count cannot be read.
    pub fn event_count(&self) -> Result<usize, StateError> {
        let count = self
            .connection
            .query_row("SELECT COUNT(*) FROM runtime_events", [], |row| row.get(0))
            .map_err(database_error)?;
        from_sql_count(count)
    }

    /// Adds or replaces one explicit memory and enforces the device-wide bound.
    ///
    /// Session memories require an expiry. Long-term memories cannot expire.
    /// Credentials and raw multiline payloads are rejected before persistence.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, capacity, or database error.
    pub fn remember(
        &mut self,
        clock: &dyn Clock,
        id: &str,
        scope: MemoryScope,
        content: &str,
        expires_at_ms: Option<u64>,
    ) -> Result<MemoryRecord, StateError> {
        validate_memory_id(id)?;
        validate_memory_content(content, self.limits.max_memory_bytes)?;
        let now_ms = clock.now_ms();
        match (scope, expires_at_ms) {
            (MemoryScope::Session, Some(expires)) if expires > now_ms => {}
            (MemoryScope::LongTerm, None) => {}
            _ => {
                return Err(StateError::new(
                    StateErrorKind::InvalidInput,
                    "memory expiry does not match its scope",
                ));
            }
        }
        self.prune_expired_memories(clock)?;
        let exists = self
            .connection
            .query_row("SELECT 1 FROM memories WHERE id = ?1", [id], |_| Ok(()))
            .optional()
            .map_err(database_error)?
            .is_some();
        if !exists && self.memory_count()? >= self.limits.max_memories {
            return Err(StateError::new(
                StateErrorKind::Capacity,
                "memory capacity reached",
            ));
        }
        let created_at_ms = if exists {
            self.connection
                .query_row(
                    "SELECT created_at_ms FROM memories WHERE id = ?1",
                    [id],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(database_error)
                .and_then(from_sql_time)?
        } else {
            now_ms
        };
        self.connection
            .execute(
                "INSERT INTO memories
                    (id, scope, content, created_at_ms, updated_at_ms, expires_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(id) DO UPDATE SET
                    scope = excluded.scope,
                    content = excluded.content,
                    updated_at_ms = excluded.updated_at_ms,
                    expires_at_ms = excluded.expires_at_ms",
                params![
                    id,
                    scope.as_str(),
                    content.trim(),
                    to_sql_time(created_at_ms)?,
                    to_sql_time(now_ms)?,
                    expires_at_ms.map(to_sql_time).transpose()?
                ],
            )
            .map_err(database_error)?;
        Ok(MemoryRecord {
            id: id.to_owned(),
            scope,
            content: content.trim().to_owned(),
            created_at_ms,
            updated_at_ms: now_ms,
            expires_at_ms,
        })
    }

    /// Returns non-expired memories in most-recently-updated order.
    ///
    /// # Errors
    ///
    /// Returns a validation or database error.
    pub fn memories(
        &mut self,
        clock: &dyn Clock,
        scope: MemoryScope,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, StateError> {
        if limit == 0 || limit > self.limits.max_memories {
            return Err(StateError::new(
                StateErrorKind::InvalidInput,
                "memory result limit is invalid",
            ));
        }
        self.prune_expired_memories(clock)?;
        let mut statement = self
            .connection
            .prepare(
                "SELECT id, scope, content, created_at_ms, updated_at_ms, expires_at_ms
                 FROM memories WHERE scope = ?1
                 ORDER BY updated_at_ms DESC, id LIMIT ?2",
            )
            .map_err(database_error)?;
        let rows = statement
            .query_map(params![scope.as_str(), to_sql_count(limit)?], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, Option<i64>>(5)?,
                ))
            })
            .map_err(database_error)?;
        rows.map(|row| {
            let (id, scope, content, created, updated, expires) = row.map_err(database_error)?;
            Ok(MemoryRecord {
                id,
                scope: MemoryScope::parse(&scope)?,
                content,
                created_at_ms: from_sql_time(created)?,
                updated_at_ms: from_sql_time(updated)?,
                expires_at_ms: expires.map(from_sql_time).transpose()?,
            })
        })
        .collect()
    }

    /// Forgets one memory by its bounded identifier.
    ///
    /// # Errors
    ///
    /// Returns not-found when no matching memory exists.
    pub fn forget_memory(&mut self, id: &str) -> Result<(), StateError> {
        validate_memory_id(id)?;
        let deleted = self
            .connection
            .execute("DELETE FROM memories WHERE id = ?1", [id])
            .map_err(database_error)?;
        if deleted == 0 {
            return Err(StateError::new(
                StateErrorKind::NotFound,
                "memory was not found",
            ));
        }
        Ok(())
    }

    fn prune_expired_memories(&mut self, clock: &dyn Clock) -> Result<(), StateError> {
        self.connection
            .execute(
                "DELETE FROM memories WHERE expires_at_ms IS NOT NULL AND expires_at_ms <= ?1",
                [to_sql_time(clock.now_ms())?],
            )
            .map_err(database_error)?;
        Ok(())
    }

    fn memory_count(&self) -> Result<usize, StateError> {
        let count = self
            .connection
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .map_err(database_error)?;
        from_sql_count(count)
    }
}

const MIGRATION_1: &str = r"
BEGIN IMMEDIATE;
CREATE TABLE runtime_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    at_ms INTEGER NOT NULL CHECK (at_ms >= 0),
    kind TEXT NOT NULL CHECK (
        kind IN ('runtime_started', 'agent_turn', 'tool_call', 'timer_scheduled', 'timer_fired', 'fault')
    ),
    outcome TEXT NOT NULL CHECK (
        outcome IN ('succeeded', 'denied', 'cancelled', 'failed')
    )
);
CREATE TABLE timers (
    id TEXT PRIMARY KEY,
    due_at_ms INTEGER NOT NULL CHECK (due_at_ms >= 0),
    created_at_ms INTEGER NOT NULL CHECK (created_at_ms >= 0),
    status TEXT NOT NULL CHECK (status IN ('scheduled', 'fired', 'cancelled'))
);
CREATE INDEX timers_due_idx ON timers(status, due_at_ms);
PRAGMA user_version = 1;
COMMIT;
";

const MIGRATION_2: &str = r"
BEGIN IMMEDIATE;
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL CHECK (scope IN ('session', 'long_term')),
    content TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL CHECK (created_at_ms >= 0),
    updated_at_ms INTEGER NOT NULL CHECK (updated_at_ms >= 0),
    expires_at_ms INTEGER CHECK (expires_at_ms IS NULL OR expires_at_ms >= 0)
);
CREATE INDEX memories_scope_updated_idx ON memories(scope, updated_at_ms DESC);
CREATE INDEX memories_expiry_idx ON memories(expires_at_ms)
    WHERE expires_at_ms IS NOT NULL;
PRAGMA user_version = 2;
COMMIT;
";

fn validate_timer_id(id: &str) -> Result<(), StateError> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(StateError::new(
            StateErrorKind::InvalidInput,
            "timer id must be a bounded ASCII identifier",
        ));
    }
    Ok(())
}

fn validate_memory_id(id: &str) -> Result<(), StateError> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(StateError::new(
            StateErrorKind::InvalidInput,
            "memory id must be a bounded ASCII identifier",
        ));
    }
    Ok(())
}

fn validate_memory_content(content: &str, max_bytes: usize) -> Result<(), StateError> {
    let trimmed = content.trim();
    let normalized = trimmed.to_ascii_lowercase();
    if trimmed.is_empty()
        || trimmed.len() > max_bytes
        || trimmed.contains(['\n', '\r', '\0'])
        || [
            "password",
            "api key",
            "api_key",
            "secret key",
            "access token",
        ]
        .iter()
        .any(|marker| normalized.contains(marker))
    {
        return Err(StateError::new(
            StateErrorKind::InvalidInput,
            "memory content is empty, unsafe, or too large",
        ));
    }
    Ok(())
}

fn to_sql_time(value: u64) -> Result<i64, StateError> {
    i64::try_from(value).map_err(|_| {
        StateError::new(
            StateErrorKind::InvalidInput,
            "timestamp is outside the supported range",
        )
    })
}

fn from_sql_time(value: i64) -> Result<u64, StateError> {
    u64::try_from(value).map_err(|_| {
        StateError::new(
            StateErrorKind::Database,
            "stored timestamp is outside the supported range",
        )
    })
}

fn to_sql_count(value: usize) -> Result<i64, StateError> {
    i64::try_from(value).map_err(|_| {
        StateError::new(
            StateErrorKind::InvalidInput,
            "state limit is outside the supported range",
        )
    })
}

fn from_sql_count(value: i64) -> Result<usize, StateError> {
    usize::try_from(value).map_err(|_| {
        StateError::new(
            StateErrorKind::Database,
            "stored count is outside the supported range",
        )
    })
}

fn database_error(_error: rusqlite::Error) -> StateError {
    StateError::new(StateErrorKind::Database, "local state database failed")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateErrorKind {
    InvalidInput,
    AlreadyExists,
    NotFound,
    Capacity,
    UnsupportedSchema,
    Database,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateError {
    pub kind: StateErrorKind,
    message: &'static str,
}

impl StateError {
    const fn new(kind: StateErrorKind, message: &'static str) -> Self {
        Self { kind, message }
    }
}

impl fmt::Display for StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl Error for StateError {}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{
        EventOutcome, ManualClock, MemoryScope, RuntimeEventKind, StateErrorKind, StateLimits,
        StateStore, TimerStatus,
    };

    fn limits() -> StateLimits {
        StateLimits {
            max_events: 3,
            max_active_timers: 2,
            max_timer_history: 2,
            max_memories: 3,
            max_memory_bytes: 80,
        }
    }

    #[test]
    fn migrations_create_current_schema() {
        let store = StateStore::in_memory(limits()).expect("store opens");
        assert_eq!(StateStore::schema_version(), 2);
        assert_eq!(store.event_count(), Ok(0));
    }

    #[test]
    fn event_history_is_bounded() {
        let mut store = StateStore::in_memory(limits()).expect("store opens");
        let clock = ManualClock::new(1_000);
        for _ in 0..5 {
            store
                .record_event(&clock, RuntimeEventKind::AgentTurn, EventOutcome::Succeeded)
                .expect("event records");
            clock.advance(Duration::from_millis(1));
        }
        assert_eq!(store.event_count(), Ok(3));
    }

    #[test]
    fn timer_survives_reopen_and_uses_manual_clock() {
        let directory = tempfile::tempdir().expect("temporary state root");
        let path = directory.path().join("oreo.db");
        let clock = ManualClock::new(1_000);
        {
            let mut store = StateStore::open(&path, limits()).expect("store opens");
            store
                .schedule_timer(&clock, "tea", 5_000)
                .expect("timer schedules");
            assert!(store.due_timers(&clock).expect("timers read").is_empty());
        }

        clock.set(5_000);
        let mut reopened = StateStore::open(&path, limits()).expect("store reopens");
        let due = reopened.due_timers(&clock).expect("timers read");
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, "tea");
        assert_eq!(due[0].status, TimerStatus::Scheduled);
        reopened.mark_timer_fired("tea").expect("timer fires");
        assert!(reopened.due_timers(&clock).expect("timers read").is_empty());
    }

    #[test]
    fn active_timer_capacity_is_enforced() {
        let mut store = StateStore::in_memory(limits()).expect("store opens");
        let clock = ManualClock::new(1_000);
        store
            .schedule_timer(&clock, "first", 2_000)
            .expect("first timer schedules");
        store
            .schedule_timer(&clock, "second", 3_000)
            .expect("second timer schedules");
        let error = store
            .schedule_timer(&clock, "third", 4_000)
            .expect_err("third timer exceeds capacity");
        assert_eq!(error.kind, StateErrorKind::Capacity);
    }

    #[test]
    fn duplicate_and_unstructured_timer_ids_are_rejected() {
        let mut store = StateStore::in_memory(limits()).expect("store opens");
        let clock = ManualClock::new(1_000);
        store
            .schedule_timer(&clock, "valid-id", 2_000)
            .expect("timer schedules");
        let duplicate = store
            .schedule_timer(&clock, "valid-id", 3_000)
            .expect_err("duplicate id is rejected");
        assert_eq!(duplicate.kind, StateErrorKind::AlreadyExists);
        let invalid = store
            .schedule_timer(&clock, "raw user prompt", 3_000)
            .expect_err("free-form timer id is rejected");
        assert_eq!(invalid.kind, StateErrorKind::InvalidInput);
    }

    #[test]
    fn terminal_timer_history_is_bounded() {
        let mut store = StateStore::in_memory(limits()).expect("store opens");
        let clock = ManualClock::new(1_000);
        for index in 0..4 {
            let id = format!("timer-{index}");
            store
                .schedule_timer(&clock, &id, 2_000 + index)
                .expect("timer schedules");
            store.mark_timer_fired(&id).expect("timer finishes");
        }

        let retained: i64 = store
            .connection
            .query_row(
                "SELECT COUNT(*) FROM timers WHERE status != 'scheduled'",
                [],
                |row| row.get(0),
            )
            .expect("history count reads");
        assert_eq!(retained, 2);
    }

    #[test]
    fn scheduled_timers_are_listed_by_deadline() {
        let mut store = StateStore::in_memory(limits()).expect("store opens");
        let clock = ManualClock::new(1_000);
        store
            .schedule_timer(&clock, "later", 4_000)
            .expect("later timer schedules");
        store
            .schedule_timer(&clock, "sooner", 2_000)
            .expect("sooner timer schedules");

        let timers = store.scheduled_timers().expect("timers list");
        assert_eq!(timers[0].id, "sooner");
        assert_eq!(timers[1].id, "later");
    }

    #[test]
    fn explicit_memories_are_bounded_and_replaceable() {
        let mut store = StateStore::in_memory(limits()).expect("store opens");
        let clock = ManualClock::new(1_000);
        store
            .remember(
                &clock,
                "tea",
                MemoryScope::LongTerm,
                "The user prefers masala tea.",
                None,
            )
            .expect("memory is stored");
        clock.advance(Duration::from_millis(10));
        let updated = store
            .remember(
                &clock,
                "tea",
                MemoryScope::LongTerm,
                "The user prefers ginger tea.",
                None,
            )
            .expect("memory is updated");
        assert_eq!(updated.created_at_ms, 1_000);
        assert_eq!(updated.updated_at_ms, 1_010);
        let memories = store
            .memories(&clock, MemoryScope::LongTerm, 3)
            .expect("memories list");
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].content, "The user prefers ginger tea.");
        store.forget_memory("tea").expect("memory is forgotten");
        assert!(
            store
                .memories(&clock, MemoryScope::LongTerm, 3)
                .expect("memories list")
                .is_empty()
        );
    }

    #[test]
    fn session_memory_expires_without_exposing_credentials() {
        let mut store = StateStore::in_memory(limits()).expect("store opens");
        let clock = ManualClock::new(1_000);
        store
            .remember(
                &clock,
                "current-task",
                MemoryScope::Session,
                "We are choosing flowers.",
                Some(2_000),
            )
            .expect("session memory is stored");
        assert!(
            store
                .remember(
                    &clock,
                    "unsafe",
                    MemoryScope::LongTerm,
                    "My API key is abc123",
                    None,
                )
                .is_err()
        );
        clock.set(2_000);
        assert!(
            store
                .memories(&clock, MemoryScope::Session, 3)
                .expect("expired memory is pruned")
                .is_empty()
        );
    }
}
