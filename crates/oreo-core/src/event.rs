#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalIntent {
    Pause,
    Stop,
    VolumeUp,
    VolumeDown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Route {
    Local(LocalIntent),
    Agent,
}

impl Route {
    #[must_use]
    pub fn classify(request: &str) -> Self {
        match request.trim().to_ascii_lowercase().as_str() {
            "pause" | "pause music" => Self::Local(LocalIntent::Pause),
            "stop" | "stop music" => Self::Local(LocalIntent::Stop),
            "volume up" => Self::Local(LocalIntent::VolumeUp),
            "volume down" => Self::Local(LocalIntent::VolumeDown),
            _ => Self::Agent,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Affect {
    #[default]
    Neutral,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HarnessEvent {
    Started,
    TextDelta(String),
    Completed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeEvent {
    Routed(Route),
    Harness(HarnessEvent),
    ResponsePlanned { affect: Affect },
    Delivered,
}

#[cfg(test)]
mod tests {
    use super::{LocalIntent, Route};

    #[test]
    fn known_commands_route_locally() {
        assert_eq!(Route::classify(" Pause "), Route::Local(LocalIntent::Pause));
        assert_eq!(
            Route::classify("volume up"),
            Route::Local(LocalIntent::VolumeUp)
        );
    }

    #[test]
    fn natural_query_routes_to_agent() {
        assert_eq!(Route::classify("What is running?"), Route::Agent);
    }
}
