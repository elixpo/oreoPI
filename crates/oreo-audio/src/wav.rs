use std::io::Read;

use oreo_core::CancellationToken;

use crate::{AudioError, AudioErrorKind, AudioFormat, AudioLimits, AudioSource, PcmChunk};

const WAV_HEADER_BYTES: usize = 12;
const CHUNK_HEADER_BYTES: usize = 8;
const PCM_FORMAT: u16 = 1;
const PCM_BITS: u16 = 16;

/// Strict, bounded PCM WAV source used by fixtures and file-based diagnostics.
pub struct WavSource {
    format: AudioFormat,
    samples: Vec<i16>,
    cursor: usize,
    samples_per_chunk: usize,
}

impl WavSource {
    /// Parses a signed 16-bit PCM WAV without retaining the encoded input.
    ///
    /// # Errors
    ///
    /// Rejects oversized, truncated, malformed, or unsupported WAV input.
    pub fn read(reader: impl Read, limits: AudioLimits) -> Result<Self, AudioError> {
        let limits = limits.validate()?;
        let absolute_max_samples = limits.max_samples(AudioFormat {
            sample_rate_hz: crate::MAX_SAMPLE_RATE_HZ,
            channels: crate::MAX_CHANNELS,
        })?;
        let max_bytes = absolute_max_samples
            .checked_mul(size_of::<i16>())
            .and_then(|bytes| bytes.checked_add(4_096))
            .ok_or_else(|| {
                AudioError::new(AudioErrorKind::InvalidConfig, "WAV limit is invalid")
            })?;
        let read_limit = u64::try_from(max_bytes + 1)
            .map_err(|_| AudioError::new(AudioErrorKind::InvalidConfig, "WAV limit is invalid"))?;
        let mut bytes = Vec::new();
        reader
            .take(read_limit)
            .read_to_end(&mut bytes)
            .map_err(|_| AudioError::new(AudioErrorKind::Input, "WAV input could not be read"))?;
        if bytes.len() > max_bytes {
            return Err(AudioError::new(
                AudioErrorKind::Capacity,
                "WAV input exceeds the capture limit",
            ));
        }
        Self::parse(&bytes, limits)
    }

    fn parse(bytes: &[u8], limits: AudioLimits) -> Result<Self, AudioError> {
        if bytes.len() < WAV_HEADER_BYTES || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(invalid_wav());
        }
        let declared_size = read_u32(bytes, 4)? as usize;
        if declared_size.checked_add(8) != Some(bytes.len()) {
            return Err(invalid_wav());
        }

        let mut cursor = WAV_HEADER_BYTES;
        let mut format = None;
        let mut data = None;
        while cursor < bytes.len() {
            let header_end = cursor
                .checked_add(CHUNK_HEADER_BYTES)
                .ok_or_else(invalid_wav)?;
            if header_end > bytes.len() {
                return Err(invalid_wav());
            }
            let id = &bytes[cursor..cursor + 4];
            let length = read_u32(bytes, cursor + 4)? as usize;
            let start = header_end;
            let end = start.checked_add(length).ok_or_else(invalid_wav)?;
            if end > bytes.len() {
                return Err(invalid_wav());
            }
            match id {
                b"fmt " => format = Some(parse_format(&bytes[start..end])?),
                b"data" => data = Some(&bytes[start..end]),
                _ => {}
            }
            cursor = end.checked_add(length % 2).ok_or_else(invalid_wav)?;
        }

        let format = format.ok_or_else(invalid_wav)?.validate()?;
        let data = data.ok_or_else(invalid_wav)?;
        if data.is_empty() || !data.len().is_multiple_of(size_of::<i16>()) {
            return Err(invalid_wav());
        }
        let sample_count = data.len() / size_of::<i16>();
        if sample_count > limits.max_samples(format)?
            || !sample_count.is_multiple_of(usize::from(format.channels))
        {
            return Err(AudioError::new(
                AudioErrorKind::Capacity,
                "WAV audio exceeds the capture limit",
            ));
        }
        let samples = data
            .chunks_exact(2)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
            .collect();
        Ok(Self {
            format,
            samples,
            cursor: 0,
            samples_per_chunk: limits.samples_per_chunk(format)?,
        })
    }
}

impl AudioSource for WavSource {
    fn format(&self) -> AudioFormat {
        self.format
    }

    fn next_chunk(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Option<PcmChunk>, AudioError> {
        if cancellation.is_cancelled() {
            return Err(AudioError::new(
                AudioErrorKind::Cancelled,
                "audio capture was cancelled",
            ));
        }
        if self.cursor == self.samples.len() {
            return Ok(None);
        }
        let end = self
            .cursor
            .saturating_add(self.samples_per_chunk)
            .min(self.samples.len());
        let chunk = PcmChunk::new(self.format, self.samples[self.cursor..end].to_vec())?;
        self.cursor = end;
        Ok(Some(chunk))
    }
}

fn parse_format(bytes: &[u8]) -> Result<AudioFormat, AudioError> {
    if bytes.len() < 16 {
        return Err(invalid_wav());
    }
    let encoding = read_u16(bytes, 0)?;
    let channels = read_u16(bytes, 2)?;
    let sample_rate_hz = read_u32(bytes, 4)?;
    let byte_rate = read_u32(bytes, 8)?;
    let block_align = read_u16(bytes, 12)?;
    let bits_per_sample = read_u16(bytes, 14)?;
    let expected_align = channels.checked_mul(PCM_BITS / 8).ok_or_else(invalid_wav)?;
    let expected_rate = sample_rate_hz
        .checked_mul(u32::from(expected_align))
        .ok_or_else(invalid_wav)?;
    if encoding != PCM_FORMAT
        || bits_per_sample != PCM_BITS
        || block_align != expected_align
        || byte_rate != expected_rate
    {
        return Err(AudioError::new(
            AudioErrorKind::UnsupportedFormat,
            "WAV format is unsupported",
        ));
    }
    Ok(AudioFormat {
        sample_rate_hz,
        channels,
    })
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, AudioError> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(invalid_wav)?
        .try_into()
        .map_err(|_| invalid_wav())?;
    Ok(u16::from_le_bytes(value))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, AudioError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(invalid_wav)?
        .try_into()
        .map_err(|_| invalid_wav())?;
    Ok(u32::from_le_bytes(value))
}

const fn invalid_wav() -> AudioError {
    AudioError::new(AudioErrorKind::Input, "WAV input is invalid")
}

#[cfg(test)]
mod tests {
    use oreo_core::CancellationToken;

    use crate::{AudioErrorKind, AudioLimits, AudioSource};

    use super::WavSource;

    fn pcm_wav(samples: &[i16]) -> Vec<u8> {
        let data_bytes = samples.len() * 2;
        let file_bytes = 36 + data_bytes;
        let mut wav = Vec::with_capacity(file_bytes + 8);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(
            &u32::try_from(file_bytes)
                .expect("fixture size")
                .to_le_bytes(),
        );
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&16_000_u32.to_le_bytes());
        wav.extend_from_slice(&32_000_u32.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(
            &u32::try_from(data_bytes)
                .expect("fixture size")
                .to_le_bytes(),
        );
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        wav
    }

    #[test]
    fn fixture_is_streamed_in_twenty_millisecond_chunks() {
        let input = pcm_wav(&vec![7; 640]);
        let mut source = WavSource::read(input.as_slice(), AudioLimits::sbc()).expect("WAV opens");
        let cancellation = CancellationToken::new();
        let first = source
            .next_chunk(&cancellation)
            .expect("chunk reads")
            .expect("first chunk exists");
        let second = source
            .next_chunk(&cancellation)
            .expect("chunk reads")
            .expect("second chunk exists");
        assert_eq!(first.samples().len(), 320);
        assert_eq!(second.samples().len(), 320);
        assert_eq!(source.next_chunk(&cancellation), Ok(None));
    }

    #[test]
    fn cancellation_discards_remaining_fixture_audio() {
        let input = pcm_wav(&vec![0; 640]);
        let mut source = WavSource::read(input.as_slice(), AudioLimits::sbc()).expect("WAV opens");
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = source
            .next_chunk(&cancellation)
            .expect_err("cancelled read fails");
        assert_eq!(error.kind, AudioErrorKind::Cancelled);
    }

    #[test]
    fn malformed_and_unsupported_wav_are_rejected() {
        let malformed = WavSource::read(&b"not a wav"[..], AudioLimits::sbc())
            .err()
            .expect("malformed WAV fails");
        assert_eq!(malformed.kind, AudioErrorKind::Input);

        let mut unsupported = pcm_wav(&[0]);
        unsupported[34..36].copy_from_slice(&8_u16.to_le_bytes());
        let error = WavSource::read(unsupported.as_slice(), AudioLimits::sbc())
            .err()
            .expect("8-bit WAV fails");
        assert_eq!(error.kind, AudioErrorKind::UnsupportedFormat);
    }
}
