use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cloneable cancellation signal shared across one runtime turn.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::CancellationToken;

    #[test]
    fn cancellation_is_shared() {
        let first = CancellationToken::new();
        let second = first.clone();
        first.cancel();
        assert!(second.is_cancelled());
    }
}

