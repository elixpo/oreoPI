//! Versioned, bounded protocol for communication with the local Oreo daemon.

use std::error::Error;
use std::fmt;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u16 = 1;
pub const MAX_MESSAGE_BYTES: usize = 16 * 1024;
pub const MAX_TIMER_DURATION_MS: u64 = 30 * 24 * 60 * 60 * 1_000;
pub const MAX_MEMORY_DURATION_MS: u64 = 24 * 60 * 60 * 1_000;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiMemoryScope {
    Session,
    LongTerm,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Status {
        version: u16,
    },
    Diagnostics {
        version: u16,
    },
    TimerSet {
        version: u16,
        id: String,
        duration_ms: u64,
    },
    TimerList {
        version: u16,
    },
    TimerCancel {
        version: u16,
        id: String,
    },
    MemoryRemember {
        version: u16,
        id: String,
        scope: ApiMemoryScope,
        content: String,
        duration_ms: Option<u64>,
    },
    MemoryList {
        version: u16,
        scope: ApiMemoryScope,
    },
    MemoryForget {
        version: u16,
        id: String,
    },
    Shutdown {
        version: u16,
    },
}

impl Request {
    #[must_use]
    pub const fn version(&self) -> u16 {
        match self {
            Self::Status { version }
            | Self::Diagnostics { version }
            | Self::TimerSet { version, .. }
            | Self::TimerList { version }
            | Self::TimerCancel { version, .. }
            | Self::MemoryRemember { version, .. }
            | Self::MemoryList { version, .. }
            | Self::MemoryForget { version, .. }
            | Self::Shutdown { version } => *version,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePhase {
    Starting,
    Ready,
    Stopping,
    Faulted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApiTimer {
    pub id: String,
    pub due_at_ms: u64,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApiMemory {
    pub id: String,
    pub scope: ApiMemoryScope,
    pub content: String,
    pub updated_at_ms: u64,
    pub expires_at_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    UnsupportedVersion,
    NotFound,
    AlreadyExists,
    Capacity,
    Internal,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Response {
    Status {
        version: u16,
        phase: RuntimePhase,
        schema_version: u32,
        active_timers: usize,
    },
    Diagnostics {
        version: u16,
        phase: RuntimePhase,
        schema_version: u32,
        retained_events: usize,
        event_limit: usize,
        active_timers: usize,
        timer_limit: usize,
        resident_memory_kib: Option<u64>,
    },
    TimerSet {
        version: u16,
        timer: ApiTimer,
    },
    TimerList {
        version: u16,
        timers: Vec<ApiTimer>,
    },
    TimerCancelled {
        version: u16,
        id: String,
    },
    MemoryRemembered {
        version: u16,
        memory: ApiMemory,
    },
    MemoryList {
        version: u16,
        memories: Vec<ApiMemory>,
    },
    MemoryForgotten {
        version: u16,
        id: String,
    },
    ShuttingDown {
        version: u16,
    },
    Error {
        version: u16,
        code: ErrorCode,
        message: String,
    },
}

/// Reads one bounded JSON message terminated by EOF or a newline.
///
/// # Errors
///
/// Returns a typed protocol failure for I/O, oversized input, malformed JSON,
/// or trailing data after the first line.
pub fn read_request(reader: &mut impl Read) -> Result<Request, LocalApiError> {
    read_json(reader)
}

/// Writes one compact JSON response followed by a newline.
///
/// # Errors
///
/// Returns a typed protocol failure for serialization or I/O failure.
pub fn write_response(writer: &mut impl Write, response: &Response) -> Result<(), LocalApiError> {
    write_json(writer, response)
}

fn read_response(reader: &mut impl Read) -> Result<Response, LocalApiError> {
    read_json(reader)
}

fn read_json<T: for<'de> Deserialize<'de>>(reader: &mut impl Read) -> Result<T, LocalApiError> {
    let limit = u64::try_from(MAX_MESSAGE_BYTES + 1).expect("message limit fits u64");
    let mut bytes = Vec::with_capacity(MAX_MESSAGE_BYTES.min(1_024));
    reader
        .take(limit)
        .read_to_end(&mut bytes)
        .map_err(|_| LocalApiError::new(LocalApiErrorKind::Io, "local API read failed"))?;
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err(LocalApiError::new(
            LocalApiErrorKind::TooLarge,
            "local API message exceeds its limit",
        ));
    }
    let trimmed = bytes.strip_suffix(b"\n").unwrap_or(&bytes);
    let trimmed = trimmed.strip_suffix(b"\r").unwrap_or(trimmed);
    if trimmed.contains(&b'\n') || trimmed.contains(&b'\r') {
        return Err(LocalApiError::new(
            LocalApiErrorKind::InvalidMessage,
            "local API accepts one message per connection",
        ));
    }
    serde_json::from_slice(trimmed).map_err(|_| {
        LocalApiError::new(
            LocalApiErrorKind::InvalidMessage,
            "local API message is invalid",
        )
    })
}

fn write_json<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<(), LocalApiError> {
    let bytes = serde_json::to_vec(value).map_err(|_| {
        LocalApiError::new(
            LocalApiErrorKind::InvalidMessage,
            "local API message could not be encoded",
        )
    })?;
    if bytes.len() + 1 > MAX_MESSAGE_BYTES {
        return Err(LocalApiError::new(
            LocalApiErrorKind::TooLarge,
            "local API message exceeds its limit",
        ));
    }
    writer
        .write_all(&bytes)
        .and_then(|()| writer.write_all(b"\n"))
        .and_then(|()| writer.flush())
        .map_err(|_| LocalApiError::new(LocalApiErrorKind::Io, "local API write failed"))
}

/// Sends one request to the user-local daemon.
///
/// # Errors
///
/// Returns unavailable on non-Unix platforms or when the socket cannot be
/// reached, and a typed protocol error for framing failures.
pub fn send_request(path: &Path, request: &Request) -> Result<Response, LocalApiError> {
    send_request_platform(path, request)
}

#[cfg(unix)]
fn send_request_platform(path: &Path, request: &Request) -> Result<Response, LocalApiError> {
    use std::net::Shutdown;
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    let mut stream = UnixStream::connect(path).map_err(|_| {
        LocalApiError::new(
            LocalApiErrorKind::Unavailable,
            "local Oreo daemon is unavailable",
        )
    })?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .and_then(|()| stream.set_write_timeout(Some(Duration::from_secs(2))))
        .map_err(|_| LocalApiError::new(LocalApiErrorKind::Io, "local API timeout setup failed"))?;
    write_json(&mut stream, request)?;
    stream
        .shutdown(Shutdown::Write)
        .map_err(|_| LocalApiError::new(LocalApiErrorKind::Io, "local API request failed"))?;
    read_response(&mut stream)
}

#[cfg(not(unix))]
fn send_request_platform(_path: &Path, _request: &Request) -> Result<Response, LocalApiError> {
    Err(LocalApiError::new(
        LocalApiErrorKind::Unavailable,
        "local Oreo daemon requires a Unix socket",
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalApiErrorKind {
    Unavailable,
    Io,
    TooLarge,
    InvalidMessage,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalApiError {
    pub kind: LocalApiErrorKind,
    message: &'static str,
}

impl LocalApiError {
    const fn new(kind: LocalApiErrorKind, message: &'static str) -> Self {
        Self { kind, message }
    }
}

impl fmt::Display for LocalApiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl Error for LocalApiError {}

#[cfg(test)]
mod tests {
    use super::{
        ApiMemoryScope, LocalApiErrorKind, MAX_MESSAGE_BYTES, PROTOCOL_VERSION, Request, Response,
        read_request, write_response,
    };

    #[test]
    fn request_round_trip_uses_one_bounded_line() {
        let request = Request::TimerSet {
            version: PROTOCOL_VERSION,
            id: "tea".to_owned(),
            duration_ms: 30_000,
        };
        let encoded = serde_json::to_vec(&request).expect("request encodes");
        assert_eq!(read_request(&mut encoded.as_slice()), Ok(request));
    }

    #[test]
    fn request_accepts_crlf_termination() {
        let input = format!("{{\"type\":\"status\",\"version\":{PROTOCOL_VERSION}}}\r\n");
        assert_eq!(
            read_request(&mut input.as_bytes()),
            Ok(Request::Status {
                version: PROTOCOL_VERSION,
            })
        );
    }

    #[test]
    fn multiple_messages_are_rejected() {
        let input = format!(
            "{{\"type\":\"status\",\"version\":{PROTOCOL_VERSION}}}\n{{\"type\":\"shutdown\",\"version\":{PROTOCOL_VERSION}}}\n"
        );
        let error = read_request(&mut input.as_bytes()).expect_err("second message is rejected");
        assert_eq!(error.kind, LocalApiErrorKind::InvalidMessage);
    }

    #[test]
    fn oversized_request_is_rejected_before_json_decode() {
        let input = vec![b'x'; MAX_MESSAGE_BYTES + 1];
        let error = read_request(&mut input.as_slice()).expect_err("oversized input is rejected");
        assert_eq!(error.kind, LocalApiErrorKind::TooLarge);
    }

    #[test]
    fn response_encoding_is_newline_terminated() {
        let mut encoded = Vec::new();
        write_response(
            &mut encoded,
            &Response::ShuttingDown {
                version: PROTOCOL_VERSION,
            },
        )
        .expect("response encodes");
        assert!(encoded.ends_with(b"\n"));
    }

    #[test]
    fn explicit_memory_request_round_trips_without_shell_parsing() {
        let request = Request::MemoryRemember {
            version: PROTOCOL_VERSION,
            id: "tea".to_owned(),
            scope: ApiMemoryScope::LongTerm,
            content: "The user prefers ginger tea.".to_owned(),
            duration_ms: None,
        };
        let encoded = serde_json::to_vec(&request).expect("request encodes");
        assert_eq!(read_request(&mut encoded.as_slice()), Ok(request));
    }
}
