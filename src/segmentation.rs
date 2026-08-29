use std::cmp::Ordering;
use std::ops::Range;

use crate::Error;

pub fn validate_min_size(min_size: usize) -> Result<(), Error> {
    if min_size == 0 {
        return Err(Error::InvalidMinSize { value: min_size });
    }
    Ok(())
}

pub fn validate_jump(jump: usize) -> Result<(), Error> {
    if jump == 0 {
        return Err(Error::InvalidJump { value: jump });
    }
    Ok(())
}

pub fn validate_penalty(penalty: f64) -> Result<(), Error> {
    if !penalty.is_finite() || penalty <= 0.0 {
        return Err(Error::InvalidPenalty { value: penalty });
    }
    Ok(())
}

pub fn validate_segment(
    segment: Range<usize>,
    n_samples: usize,
    min_size: usize,
) -> Result<(), Error> {
    validate_min_size(min_size)?;
    if segment.start >= segment.end || segment.end > n_samples {
        return Err(Error::InvalidRange {
            start: segment.start,
            end: segment.end,
            n_samples,
        });
    }
    let length = segment.end - segment.start;
    if length < min_size {
        return Err(Error::SegmentTooShort {
            start: segment.start,
            end: segment.end,
            length,
            minimum: min_size,
        });
    }
    Ok(())
}

pub fn validate_breakpoints(
    breakpoints: &[usize],
    n_samples: usize,
    min_size: usize,
) -> Result<(), Error> {
    validate_min_size(min_size)?;
    if breakpoints.is_empty() {
        return Err(Error::EmptyBreakpoints);
    }

    let mut start = 0;
    for (position, &end) in breakpoints.iter().enumerate() {
        if end <= start || end > n_samples {
            return Err(Error::InvalidBreakpoint {
                position,
                value: end,
                n_samples,
            });
        }
        validate_segment(start..end, n_samples, min_size)?;
        start = end;
    }

    if start != n_samples {
        return Err(Error::MissingTerminalBreakpoint {
            actual: start,
            n_samples,
        });
    }
    Ok(())
}

pub fn partition_cost<F>(breakpoints: &[usize], mut segment_cost: F) -> Result<f64, Error>
where
    F: FnMut(Range<usize>) -> f64,
{
    let mut start = 0;
    let mut total = 0.0;
    for &end in breakpoints {
        let value = segment_cost(start..end);
        if !value.is_finite() {
            return Err(Error::NonFiniteObjective { value });
        }
        total += value;
        start = end;
    }
    if !total.is_finite() {
        return Err(Error::NonFiniteObjective { value: total });
    }
    Ok(total)
}

pub fn candidate_is_better(
    candidate_cost: f64,
    candidate_breakpoints: &[usize],
    best_cost: f64,
    best_breakpoints: &[usize],
) -> bool {
    match candidate_cost.total_cmp(&best_cost) {
        Ordering::Less => true,
        Ordering::Equal => candidate_breakpoints < best_breakpoints,
        Ordering::Greater => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_breakpoints_with_terminal_sample() {
        assert_eq!(validate_breakpoints(&[2, 5], 5, 2), Ok(()));
    }

    #[test]
    fn rejects_missing_terminal_breakpoint() {
        assert_eq!(
            validate_breakpoints(&[2, 4], 5, 2),
            Err(Error::MissingTerminalBreakpoint {
                actual: 4,
                n_samples: 5,
            })
        );
    }

    #[test]
    fn rejects_short_segments() {
        assert!(matches!(
            validate_breakpoints(&[1, 5], 5, 2),
            Err(Error::SegmentTooShort { .. })
        ));
    }

    #[test]
    fn tie_breaks_lexicographically() {
        assert!(candidate_is_better(3.0, &[2, 5], 3.0, &[3, 5]));
        assert!(!candidate_is_better(3.0, &[3, 5], 3.0, &[2, 5]));
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert_eq!(
            validate_min_size(0),
            Err(Error::InvalidMinSize { value: 0 })
        );
        assert_eq!(validate_jump(0), Err(Error::InvalidJump { value: 0 }));
        assert!(matches!(
            validate_penalty(f64::NAN),
            Err(Error::InvalidPenalty { .. })
        ));
    }
}
