use crate::{AudioError, AudioErrorKind, AudioLimits};

/// Converts machine-oriented text into deterministic, bounded TTS input.
///
/// # Errors
///
/// Rejects empty input or input/output that exceeds the response-buffer bound.
pub fn normalize_for_speech(text: &str, limits: AudioLimits) -> Result<String, AudioError> {
    let limits = limits.validate()?;
    let text = text.trim();
    if text.is_empty() {
        return Err(AudioError::new(
            AudioErrorKind::NoSpeech,
            "speech text is empty",
        ));
    }
    if text.len() > limits.max_response_buffer_bytes {
        return Err(capacity_error());
    }

    let mut output = String::new();
    for token in text.split_whitespace() {
        let normalized = normalize_token(token);
        let separator = usize::from(!output.is_empty());
        if output
            .len()
            .saturating_add(separator)
            .saturating_add(normalized.len())
            > limits.max_response_buffer_bytes
        {
            return Err(capacity_error());
        }
        if separator != 0 {
            output.push(' ');
        }
        output.push_str(&normalized);
    }
    Ok(output)
}

fn normalize_token(token: &str) -> String {
    let (core, suffix) = split_sentence_suffix(token);
    let normalized = if core.starts_with("http://") || core.starts_with("https://") {
        pronounce_url(core)
    } else if is_ipv4(core) {
        pronounce_separated(core, '.')
    } else if is_commit_sha(core) {
        pronounce_characters(core)
    } else if let Some((number, unit)) = split_number_unit(core) {
        format!(
            "{} {}",
            pronounce_number(number),
            unit_name(unit, is_one(number)).expect("recognized unit has a name")
        )
    } else if is_number(core) {
        pronounce_number(core)
    } else if is_acronym(core) {
        pronounce_characters(core)
    } else {
        core.to_owned()
    };
    format!("{normalized}{suffix}")
}

fn split_sentence_suffix(token: &str) -> (&str, &str) {
    let core = token.trim_end_matches([',', ';', '!', '?', '.']);
    (&token[..core.len()], &token[core.len()..])
}

fn is_ipv4(value: &str) -> bool {
    let parts = value.split('.').collect::<Vec<_>>();
    parts.len() == 4
        && parts.iter().all(|part| {
            !part.is_empty()
                && part.chars().all(|character| character.is_ascii_digit())
                && part.parse::<u8>().is_ok()
        })
}

fn is_commit_sha(value: &str) -> bool {
    (7..=40).contains(&value.len())
        && value.chars().all(|character| character.is_ascii_hexdigit())
        && value
            .chars()
            .any(|character| character.is_ascii_alphabetic())
}

fn is_acronym(value: &str) -> bool {
    (2..=10).contains(&value.len())
        && value
            .chars()
            .all(|character| character.is_ascii_uppercase())
}

fn is_number(value: &str) -> bool {
    let unsigned = value
        .strip_prefix('-')
        .or_else(|| value.strip_prefix('+'))
        .unwrap_or(value);
    let mut parts = unsigned.split('.');
    let Some(integer) = parts.next() else {
        return false;
    };
    !integer.is_empty()
        && integer.chars().all(|character| character.is_ascii_digit())
        && parts.next().is_none_or(|fraction| {
            !fraction.is_empty() && fraction.chars().all(|character| character.is_ascii_digit())
        })
        && parts.next().is_none()
}

fn split_number_unit(value: &str) -> Option<(&str, &str)> {
    let split = value.char_indices().find_map(|(index, character)| {
        (!character.is_ascii_digit() && !matches!(character, '-' | '+' | '.')).then_some(index)
    })?;
    let (number, unit) = value.split_at(split);
    (is_number(number) && unit_name(unit, false).is_some()).then_some((number, unit))
}

fn unit_name(unit: &str, singular: bool) -> Option<&'static str> {
    Some(match (unit, singular) {
        ("kg", true) => "kilogram",
        ("kg", false) => "kilograms",
        ("g", true) => "gram",
        ("g", false) => "grams",
        ("km", true) => "kilometer",
        ("km", false) => "kilometers",
        ("m", true) => "meter",
        ("m", false) => "meters",
        ("cm", true) => "centimeter",
        ("cm", false) => "centimeters",
        ("mm", true) => "millimeter",
        ("mm", false) => "millimeters",
        ("ms", true) => "millisecond",
        ("ms", false) => "milliseconds",
        ("s", true) => "second",
        ("s", false) => "seconds",
        ("MB", true) => "megabyte",
        ("MB", false) => "megabytes",
        ("GB", true) => "gigabyte",
        ("GB", false) => "gigabytes",
        ("°C", _) => "degrees Celsius",
        ("°F", _) => "degrees Fahrenheit",
        ("%", _) => "percent",
        _ => return None,
    })
}

fn is_one(number: &str) -> bool {
    let number = number.strip_prefix('+').unwrap_or(number);
    let mut parts = number.split('.');
    parts.next() == Some("1")
        && parts
            .next()
            .is_none_or(|fraction| fraction.chars().all(|character| character == '0'))
}

fn pronounce_number(number: &str) -> String {
    let (negative, unsigned) = number
        .strip_prefix('-')
        .map_or((false, number), |unsigned| (true, unsigned));
    let unsigned = unsigned.strip_prefix('+').unwrap_or(unsigned);
    let mut parts = unsigned.split('.');
    let integer = parts.next().unwrap_or_default();
    let mut spoken = integer
        .parse::<u64>()
        .ok()
        .filter(|value| *value <= 999_999_999_999)
        .map_or_else(|| pronounce_digits(integer), integer_words);
    if let Some(fraction) = parts.next() {
        spoken.push_str(" point ");
        spoken.push_str(&pronounce_digits(fraction));
    }
    if negative {
        spoken.insert_str(0, "negative ");
    }
    spoken
}

fn integer_words(value: u64) -> String {
    if value == 0 {
        return "zero".to_owned();
    }
    let groups = [
        (1_000_000_000_u64, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ];
    let mut remaining = value;
    let mut words = Vec::new();
    for (size, name) in groups {
        if remaining >= size {
            words.push(under_thousand(remaining / size));
            words.push(name.to_owned());
            remaining %= size;
        }
    }
    if remaining != 0 {
        words.push(under_thousand(remaining));
    }
    words.join(" ")
}

fn under_thousand(value: u64) -> String {
    const SMALL: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    let mut remaining = value;
    let mut words = Vec::new();
    if remaining >= 100 {
        words.push(SMALL[usize::try_from(remaining / 100).expect("digit index")]);
        words.push("hundred");
        remaining %= 100;
    }
    if remaining >= 20 {
        words.push(TENS[usize::try_from(remaining / 10).expect("tens index")]);
        remaining %= 10;
    }
    if remaining != 0 {
        words.push(SMALL[usize::try_from(remaining).expect("small number index")]);
    }
    words.join(" ")
}

fn pronounce_digits(value: &str) -> String {
    value
        .chars()
        .filter_map(|character| character.to_digit(10))
        .map(|digit| digit_word(u8::try_from(digit).expect("decimal digit")))
        .collect::<Vec<_>>()
        .join(" ")
}

fn pronounce_characters(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            character.to_digit(10).map_or_else(
                || character.to_string(),
                |digit| digit_word(u8::try_from(digit).expect("decimal digit")).to_owned(),
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn pronounce_separated(value: &str, separator: char) -> String {
    value
        .split(separator)
        .map(pronounce_digits)
        .collect::<Vec<_>>()
        .join(" dot ")
}

fn pronounce_url(value: &str) -> String {
    let without_scheme = value
        .strip_prefix("https://")
        .or_else(|| value.strip_prefix("http://"))
        .unwrap_or(value);
    let value = without_scheme
        .strip_prefix("www.")
        .unwrap_or(without_scheme);
    let mut spoken = String::new();
    for character in value.chars() {
        let replacement = match character {
            '.' => " dot ",
            '/' => " slash ",
            '-' => " dash ",
            '_' => " underscore ",
            ':' => " colon ",
            '?' => " question mark ",
            '&' => " and ",
            '=' => " equals ",
            '%' => " percent ",
            _ => {
                spoken.push(character);
                continue;
            }
        };
        spoken.push_str(replacement);
    }
    spoken.split_whitespace().collect::<Vec<_>>().join(" ")
}

const fn digit_word(digit: u8) -> &'static str {
    match digit {
        0 => "zero",
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        6 => "six",
        7 => "seven",
        8 => "eight",
        _ => "nine",
    }
}

const fn capacity_error() -> AudioError {
    AudioError::new(AudioErrorKind::Capacity, "speech text exceeds its limit")
}

#[cfg(test)]
mod tests {
    use crate::{AudioErrorKind, AudioLimits};

    use super::normalize_for_speech;

    fn normalize(text: &str) -> String {
        normalize_for_speech(text, AudioLimits::sbc()).expect("text normalizes")
    }

    #[test]
    fn numbers_and_units_are_spoken_deterministically() {
        assert_eq!(
            normalize("Set 12kg for 1.5km at -3°C."),
            "Set twelve kilograms for one point five kilometers at negative three degrees Celsius."
        );
        assert_eq!(
            normalize("Wait 1s or 20ms."),
            "Wait one second or twenty milliseconds."
        );
    }

    #[test]
    fn acronyms_urls_ips_and_shas_are_expanded() {
        assert_eq!(
            normalize("API https://example.com/v1 192.168.1.2 a1b2c3d"),
            "A P I example dot com slash v1 one nine two dot one six eight dot one dot two a one b two c three d"
        );
    }

    #[test]
    fn normalized_output_cannot_exceed_response_bound() {
        let limits = AudioLimits {
            max_response_buffer_bytes: 64,
            ..AudioLimits::sbc()
        };
        let error = normalize_for_speech("API API API API API API API API API API API API", limits)
            .expect_err("expansion is bounded");
        assert_eq!(error.kind, AudioErrorKind::Capacity);
    }
}
