use std::collections::VecDeque;
use std::time::Duration;

use crate::{AudioError, AudioErrorKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LatencyMetric {
    Endpoint,
    CachedStt,
    TtsFirstAudio,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatencyTargets {
    pub endpoint_ms_exclusive: u32,
    pub cached_stt_ms_exclusive: u32,
    pub tts_first_audio_ms_exclusive: u32,
}

impl LatencyTargets {
    #[must_use]
    pub const fn wp004() -> Self {
        Self {
            endpoint_ms_exclusive: 500,
            cached_stt_ms_exclusive: 1_200,
            tts_first_audio_ms_exclusive: 500,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatencyReport {
    pub sample_count: usize,
    pub endpoint_p95_ms: Option<u32>,
    pub cached_stt_p95_ms: Option<u32>,
    pub tts_first_audio_p95_ms: Option<u32>,
}

impl LatencyReport {
    #[must_use]
    pub fn meets(self, targets: LatencyTargets) -> bool {
        self.sample_count > 0
            && self
                .endpoint_p95_ms
                .is_some_and(|value| value < targets.endpoint_ms_exclusive)
            && self
                .cached_stt_p95_ms
                .is_some_and(|value| value < targets.cached_stt_ms_exclusive)
            && self
                .tts_first_audio_p95_ms
                .is_some_and(|value| value < targets.tts_first_audio_ms_exclusive)
    }
}

/// Bounded rolling latency samples used by laptop and SBC qualification.
pub struct LatencyWindow {
    capacity: usize,
    endpoint_ms: VecDeque<u32>,
    cached_stt_ms: VecDeque<u32>,
    tts_first_audio_ms: VecDeque<u32>,
}

impl LatencyWindow {
    /// Creates a window retaining at most 10,000 samples per metric.
    ///
    /// # Errors
    ///
    /// Rejects a zero or excessive capacity.
    pub fn new(capacity: usize) -> Result<Self, AudioError> {
        if !(1..=10_000).contains(&capacity) {
            return Err(AudioError::new(
                AudioErrorKind::InvalidConfig,
                "latency window capacity is invalid",
            ));
        }
        Ok(Self {
            capacity,
            endpoint_ms: VecDeque::with_capacity(capacity),
            cached_stt_ms: VecDeque::with_capacity(capacity),
            tts_first_audio_ms: VecDeque::with_capacity(capacity),
        })
    }

    pub fn record(&mut self, metric: LatencyMetric, duration: Duration) {
        let millis = u32::try_from(duration.as_millis()).unwrap_or(u32::MAX);
        let capacity = self.capacity;
        let samples = match metric {
            LatencyMetric::Endpoint => &mut self.endpoint_ms,
            LatencyMetric::CachedStt => &mut self.cached_stt_ms,
            LatencyMetric::TtsFirstAudio => &mut self.tts_first_audio_ms,
        };
        if samples.len() == capacity {
            samples.pop_front();
        }
        samples.push_back(millis);
    }

    #[must_use]
    pub fn report(&self) -> LatencyReport {
        LatencyReport {
            sample_count: self
                .endpoint_ms
                .len()
                .min(self.cached_stt_ms.len())
                .min(self.tts_first_audio_ms.len()),
            endpoint_p95_ms: p95(&self.endpoint_ms),
            cached_stt_p95_ms: p95(&self.cached_stt_ms),
            tts_first_audio_p95_ms: p95(&self.tts_first_audio_ms),
        }
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

fn p95(samples: &VecDeque<u32>) -> Option<u32> {
    if samples.is_empty() {
        return None;
    }
    let mut ordered = samples.iter().copied().collect::<Vec<_>>();
    ordered.sort_unstable();
    let rank = ordered.len().saturating_mul(95).div_ceil(100);
    ordered.get(rank.saturating_sub(1)).copied()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{LatencyMetric, LatencyTargets, LatencyWindow};

    #[test]
    fn p95_uses_nearest_rank_and_matches_wp004_gates() {
        let mut window = LatencyWindow::new(100).expect("window starts");
        for index in 0..100 {
            window.record(LatencyMetric::Endpoint, Duration::from_millis(300));
            window.record(LatencyMetric::CachedStt, Duration::from_millis(800));
            let tts = if index < 95 { 450 } else { 900 };
            window.record(LatencyMetric::TtsFirstAudio, Duration::from_millis(tts));
        }
        let report = window.report();
        assert_eq!(report.endpoint_p95_ms, Some(300));
        assert_eq!(report.cached_stt_p95_ms, Some(800));
        assert_eq!(report.tts_first_audio_p95_ms, Some(450));
        assert!(report.meets(LatencyTargets::wp004()));
    }

    #[test]
    fn exact_exclusive_limit_fails_gate() {
        let mut window = LatencyWindow::new(1).expect("window starts");
        window.record(LatencyMetric::Endpoint, Duration::from_millis(500));
        window.record(LatencyMetric::CachedStt, Duration::from_millis(1_199));
        window.record(LatencyMetric::TtsFirstAudio, Duration::from_millis(499));
        assert!(!window.report().meets(LatencyTargets::wp004()));
    }

    #[test]
    fn rolling_window_evicts_oldest_samples() {
        let mut window = LatencyWindow::new(2).expect("window starts");
        window.record(LatencyMetric::Endpoint, Duration::from_millis(900));
        window.record(LatencyMetric::Endpoint, Duration::from_millis(400));
        window.record(LatencyMetric::Endpoint, Duration::from_millis(300));
        assert_eq!(window.capacity(), 2);
        assert_eq!(window.report().endpoint_p95_ms, Some(400));
    }

    #[test]
    fn incomplete_metrics_cannot_pass() {
        let mut window = LatencyWindow::new(1).expect("window starts");
        window.record(LatencyMetric::Endpoint, Duration::from_millis(300));
        assert!(!window.report().meets(LatencyTargets::wp004()));
    }
}
