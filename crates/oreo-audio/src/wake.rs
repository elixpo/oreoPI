use std::collections::{HashMap, VecDeque};

use oreo_core::CancellationToken;

use crate::{AudioError, AudioErrorKind, AudioFormat, PcmChunk, STT_FORMAT};

const MAX_WAKE_TEXT_BYTES: usize = 512;
const WAKE_INTENT_THRESHOLD: f64 = 5.0;
const EMBEDDED_CORPUS: &str = include_str!("../../../config/wake-intent-corpus.tsv");

/// Bounded rolling PCM retained only while listening for a possible wake intent.
pub struct WakeAudioWindow {
    samples: VecDeque<i16>,
    max_samples: usize,
}

impl WakeAudioWindow {
    /// Creates a one-to-three second 16 kHz mono rolling window.
    ///
    /// # Errors
    ///
    /// Rejects durations outside the privacy and memory bound.
    pub fn new(seconds: u8) -> Result<Self, AudioError> {
        if !(1..=3).contains(&seconds) {
            return Err(AudioError::new(
                AudioErrorKind::InvalidConfig,
                "wake audio window must be one to three seconds",
            ));
        }
        Ok(Self {
            samples: VecDeque::with_capacity(usize::from(seconds) * 16_000),
            max_samples: usize::from(seconds) * 16_000,
        })
    }

    /// Adds one transient STT-format chunk and forgets the oldest samples.
    ///
    /// # Errors
    ///
    /// Rejects format changes and cancellation.
    pub fn push(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<(), AudioError> {
        if cancellation.is_cancelled() {
            self.clear();
            return Err(AudioError::new(
                AudioErrorKind::Cancelled,
                "wake listening was cancelled",
            ));
        }
        if chunk.format() != STT_FORMAT {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "wake audio requires 16 kHz mono PCM",
            ));
        }
        self.samples.extend(chunk.samples());
        let excess = self.samples.len().saturating_sub(self.max_samples);
        self.samples.drain(..excess);
        Ok(())
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<i16> {
        self.samples.iter().copied().collect()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        u64::try_from(self.samples.len()).unwrap_or(u64::MAX) * 1_000 / 16_000
    }

    #[must_use]
    pub const fn format(&self) -> AudioFormat {
        STT_FORMAT
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WakeIntentDecision {
    pub addressed: bool,
    pub score: f64,
}

/// Tiny multinomial Naive Bayes classifier trained from the reviewed corpus.
pub struct WakeIntentClassifier {
    weights: HashMap<String, [u32; 2]>,
    feature_totals: [u32; 2],
    document_totals: [u32; 2],
}

impl WakeIntentClassifier {
    /// Trains the embedded, version-controlled wake-context model in memory.
    ///
    /// # Errors
    ///
    /// Fails closed if the embedded corpus is malformed or incomplete.
    pub fn embedded() -> Result<Self, AudioError> {
        Self::train(EMBEDDED_CORPUS)
    }

    fn train(corpus: &str) -> Result<Self, AudioError> {
        let mut classifier = Self {
            weights: HashMap::new(),
            feature_totals: [0; 2],
            document_totals: [0; 2],
        };
        for line in corpus.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<_> = line.splitn(3, '\t').collect();
            let [split, label, text] = fields.as_slice() else {
                return Err(classifier_error());
            };
            if *split != "train" {
                continue;
            }
            let class = match *label {
                "ignore" => 0,
                "wake" => 1,
                _ => return Err(classifier_error()),
            };
            classifier.document_totals[class] += 1;
            for feature in features(text) {
                classifier.weights.entry(feature).or_insert([0; 2])[class] += 1;
                classifier.feature_totals[class] += 1;
            }
        }
        if classifier.document_totals.iter().any(|count| *count < 10)
            || classifier.weights.is_empty()
        {
            return Err(classifier_error());
        }
        Ok(classifier)
    }

    /// Scores whether a candidate transcript addresses Oreo.
    ///
    /// # Errors
    ///
    /// Rejects empty or oversized transcripts.
    pub fn classify(&self, transcript: &str) -> Result<WakeIntentDecision, AudioError> {
        if transcript.trim().is_empty() || transcript.len() > MAX_WAKE_TEXT_BYTES {
            return Err(AudioError::new(
                AudioErrorKind::Capacity,
                "wake transcript is empty or too large",
            ));
        }
        let vocabulary = f64::from(u32::try_from(self.weights.len()).unwrap_or(u32::MAX));
        let documents = f64::from(self.document_totals[0] + self.document_totals[1]);
        let mut scores = [0.0_f64; 2];
        for (class, score) in scores.iter_mut().enumerate() {
            *score = ((f64::from(self.document_totals[class]) + 1.0) / (documents + 2.0)).ln();
        }
        for feature in features(transcript) {
            let counts = self.weights.get(&feature).copied().unwrap_or([0; 2]);
            for (class, score) in scores.iter_mut().enumerate() {
                *score += ((f64::from(counts[class]) + 1.0)
                    / (f64::from(self.feature_totals[class]) + vocabulary))
                    .ln();
            }
        }
        let score = scores[1] - scores[0];
        Ok(WakeIntentDecision {
            addressed: score >= WAKE_INTENT_THRESHOLD,
            score,
        })
    }
}

fn features(text: &str) -> Vec<String> {
    let normalized: String = text
        .chars()
        .flat_map(char::to_lowercase)
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let words: Vec<&str> = normalized.split_whitespace().collect();
    let mut output = Vec::new();
    if let Some(first) = words.first() {
        output.push(format!("first:{first}"));
    }
    if let Some(last) = words.last() {
        output.push(format!("last:{last}"));
    }
    for word in &words {
        output.push(format!("w:{word}"));
    }
    for pair in words.windows(2) {
        output.push(format!("b:{}_{}", pair[0], pair[1]));
    }
    let padded = format!("  {normalized}  ");
    let characters: Vec<char> = padded.chars().collect();
    for gram in characters.windows(3) {
        output.push(format!("c:{}{}{}", gram[0], gram[1], gram[2]));
    }
    output
}

fn classifier_error() -> AudioError {
    AudioError::new(
        AudioErrorKind::InvalidConfig,
        "wake intent corpus is invalid",
    )
}

#[cfg(test)]
mod tests {
    use oreo_core::CancellationToken;

    use crate::{AudioErrorKind, AudioFormat, PcmChunk, STT_FORMAT};

    use super::{WakeAudioWindow, WakeIntentClassifier};

    #[test]
    fn rolling_audio_window_forgets_oldest_samples() {
        let mut window = WakeAudioWindow::new(1).expect("window is valid");
        let cancellation = CancellationToken::new();
        for value in 0..11_i16 {
            window
                .push(
                    &PcmChunk::new(STT_FORMAT, vec![value; 1_600]).expect("chunk is valid"),
                    &cancellation,
                )
                .expect("chunk enters window");
        }
        let samples = window.snapshot();
        assert_eq!(samples.len(), 16_000);
        assert_eq!(samples[0], 1);
        assert_eq!(samples[15_999], 10);
        assert_eq!(window.duration_ms(), 1_000);
    }

    #[test]
    fn rolling_window_rejects_format_changes_and_clears_on_cancel() {
        let mut window = WakeAudioWindow::new(2).expect("window is valid");
        let cancellation = CancellationToken::new();
        let wrong = PcmChunk::new(
            AudioFormat {
                sample_rate_hz: 24_000,
                channels: 1,
            },
            vec![0; 480],
        )
        .expect("chunk is valid");
        assert_eq!(
            window
                .push(&wrong, &cancellation)
                .expect_err("format fails")
                .kind,
            AudioErrorKind::UnsupportedFormat
        );
        window
            .push(
                &PcmChunk::new(STT_FORMAT, vec![0; 320]).expect("chunk is valid"),
                &cancellation,
            )
            .expect("chunk enters window");
        cancellation.cancel();
        assert_eq!(
            window
                .push(
                    &PcmChunk::new(STT_FORMAT, vec![0; 320]).expect("chunk is valid"),
                    &cancellation,
                )
                .expect_err("cancellation fails")
                .kind,
            AudioErrorKind::Cancelled
        );
        assert!(window.snapshot().is_empty());
    }

    #[test]
    fn embedded_classifier_separates_addressing_from_mentions() {
        let classifier = WakeIntentClassifier::embedded().expect("classifier trains");
        for positive in [
            "Oreo, could you listen to me?",
            "Wake up, Orio.",
            "Oreo, what time is it?",
        ] {
            assert!(
                classifier
                    .classify(positive)
                    .expect("phrase scores")
                    .addressed,
                "expected wake intent: {positive}"
            );
        }
        for negative in [
            "Please buy a packet of Oreos.",
            "Oreo cookies are discounted today.",
            "Turn down the stereo, please.",
        ] {
            let decision = classifier.classify(negative).expect("phrase scores");
            assert!(
                !decision.addressed,
                "expected ordinary mention: {negative} (score {})",
                decision.score
            );
        }
    }

    #[test]
    fn embedded_classifier_passes_reviewed_holdout_corpus() {
        let classifier = WakeIntentClassifier::embedded().expect("classifier trains");
        for line in super::EMBEDDED_CORPUS.lines() {
            let fields: Vec<_> = line.splitn(3, '\t').collect();
            if fields.first() != Some(&"test") {
                continue;
            }
            let expected = fields[1] == "wake";
            let decision = classifier.classify(fields[2]).expect("phrase scores");
            assert_eq!(
                decision.addressed, expected,
                "unexpected decision for {:?} (score {})",
                fields[2], decision.score
            );
        }
    }
}
