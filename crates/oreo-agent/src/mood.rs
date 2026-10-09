//! Bounded affect state used only to shape presentation.

/// Local affect estimate. Values are integers to keep updates deterministic on
/// small devices: valence is -100..=100; arousal and confidence are 0..=100.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AffectState {
    valence: i8,
    arousal: u8,
    confidence: u8,
}

impl AffectState {
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            valence: 0,
            arousal: 40,
            confidence: 0,
        }
    }

    /// Creates a clamped affect observation without accepting unbounded model
    /// values.
    #[must_use]
    pub fn observed(valence: i16, arousal: i16, confidence: i16) -> Self {
        Self {
            valence: i8::try_from(valence.clamp(-100, 100)).unwrap_or(0),
            arousal: bounded_u8(arousal, 0, 100),
            confidence: bounded_u8(confidence, 0, 100),
        }
    }

    /// Smooths a new observation with a fixed 3:1 prior-to-observation ratio.
    #[must_use]
    pub fn update(self, observation: Self) -> Self {
        Self::observed(
            (i16::from(self.valence) * 3 + i16::from(observation.valence)) / 4,
            (i16::from(self.arousal) * 3 + i16::from(observation.arousal)) / 4,
            i16::from(observation.confidence),
        )
    }

    /// Produces conservative speech controls. Affect never changes facts,
    /// permissions, capability selection, or safety policy.
    #[must_use]
    pub fn delivery(self) -> DeliveryStyle {
        let confidence = i16::from(self.confidence);
        let arousal_delta = (i16::from(self.arousal) - 40) * confidence / 100;
        let valence_delta = i16::from(self.valence) * confidence / 100;
        DeliveryStyle {
            rate_percent: bounded_u8(100 + arousal_delta / 5, 90, 110),
            pitch_percent: bounded_u8(100 + valence_delta / 10, 95, 105),
            warmth: bounded_u8(50 + valence_delta / 2, 20, 80),
        }
    }
}

fn bounded_u8(value: i16, minimum: i16, maximum: i16) -> u8 {
    u8::try_from(value.clamp(minimum, maximum)).unwrap_or(0)
}

/// Presentation-only controls consumable by future TTS backends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeliveryStyle {
    pub rate_percent: u8,
    pub pitch_percent: u8,
    pub warmth: u8,
}

#[cfg(test)]
mod tests {
    use super::AffectState;

    #[test]
    fn affect_is_clamped_smoothed_and_presentation_only() {
        let observation = AffectState::observed(500, -20, 120);
        let state = AffectState::neutral().update(observation);
        let delivery = state.delivery();
        assert!((90..=110).contains(&delivery.rate_percent));
        assert!((95..=105).contains(&delivery.pitch_percent));
        assert!((20..=80).contains(&delivery.warmth));
    }
}
