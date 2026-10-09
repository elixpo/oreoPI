use std::collections::HashSet;

use crate::{AudioError, AudioErrorKind};

const EMBEDDED_ADDRESSES: &str = include_str!("../../../config/conversation-addresses.txt");
const EMBEDDED_SLEEP_PHRASES: &str = include_str!("../../../config/conversation-sleep-phrases.txt");
const MAX_CONTEXT_TEXT_BYTES: usize = 4_096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConversationDirective {
    Command,
    AddressOnly,
    Sleep,
}

/// Classifies small, deterministic conversation controls without an LLM call.
///
/// Addresses only have meaning after the acoustic wake path has opened a
/// conversation. They are deliberately not added to the always-on wake model.
pub struct ConversationLanguage {
    addresses: HashSet<String>,
    sleep_phrases: HashSet<String>,
}

impl ConversationLanguage {
    /// Loads the reviewed, version-controlled conversation vocabulary.
    ///
    /// # Errors
    ///
    /// Rejects empty or unreasonably large embedded vocabularies.
    pub fn embedded() -> Result<Self, AudioError> {
        let addresses = parse_phrases(EMBEDDED_ADDRESSES);
        let sleep_phrases = parse_phrases(EMBEDDED_SLEEP_PHRASES);
        if addresses.is_empty()
            || sleep_phrases.is_empty()
            || addresses.len() > 32
            || sleep_phrases.len() > 32
        {
            return Err(AudioError::new(
                AudioErrorKind::InvalidConfig,
                "conversation vocabulary is invalid",
            ));
        }
        Ok(Self {
            addresses,
            sleep_phrases,
        })
    }

    /// Interprets only local session controls; all other text remains a command.
    ///
    /// # Errors
    ///
    /// Rejects empty or oversized transcripts.
    pub fn classify(&self, transcript: &str) -> Result<ConversationDirective, AudioError> {
        if transcript.trim().is_empty() || transcript.len() > MAX_CONTEXT_TEXT_BYTES {
            return Err(AudioError::new(
                AudioErrorKind::Capacity,
                "conversation transcript is empty or too large",
            ));
        }
        let normalized = normalize(transcript);
        if self.sleep_phrases.contains(&normalized) {
            return Ok(ConversationDirective::Sleep);
        }
        let address = strip_vocative(&normalized);
        if self.addresses.contains(address) {
            return Ok(ConversationDirective::AddressOnly);
        }
        Ok(ConversationDirective::Command)
    }
}

fn parse_phrases(source: &str) -> HashSet<String> {
    source
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(normalize)
        .collect()
}

fn normalize(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '\'' {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn strip_vocative(text: &str) -> &str {
    ["hey ", "yo ", "hello ", "okay ", "ok "]
        .iter()
        .find_map(|prefix| text.strip_prefix(prefix))
        .unwrap_or(text)
}

#[cfg(test)]
mod tests {
    use super::{ConversationDirective, ConversationLanguage};

    #[test]
    fn warm_nicknames_are_address_only() {
        let language = ConversationLanguage::embedded().expect("vocabulary loads");
        for transcript in ["mah boy", "Hey, my man!", "yo buddy", "okay boss"] {
            assert_eq!(
                language.classify(transcript).expect("text is valid"),
                ConversationDirective::AddressOnly
            );
        }
    }

    #[test]
    fn nickname_commands_remain_commands() {
        let language = ConversationLanguage::embedded().expect("vocabulary loads");
        assert_eq!(
            language
                .classify("my man, dim the lights")
                .expect("text is valid"),
            ConversationDirective::Command
        );
    }

    #[test]
    fn explicit_sleep_commands_close_the_session() {
        let language = ConversationLanguage::embedded().expect("vocabulary loads");
        assert_eq!(
            language.classify("We're done.").expect("text is valid"),
            ConversationDirective::Sleep
        );
        assert_eq!(
            language
                .classify("explain why we are done")
                .expect("text is valid"),
            ConversationDirective::Command
        );
    }
}
