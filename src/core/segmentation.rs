use std::cmp::Ordering;
use std::ops::Range;

use crate::{Error, SegmentCost};

pub(crate) const SCORE_ABSOLUTE_TOLERANCE: f64 = 1.0e-12;
pub(crate) const SCORE_RELATIVE_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchGrid {
    pub min_size: usize,
    pub jump: usize,
}

impl SearchGrid {
    pub fn new(min_size: usize, jump: usize) -> Result<Self, Error> {
        validate_min_size(min_size)?;
        validate_jump(jump)?;
        Ok(Self { min_size, jump })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stop {
    Changes(usize),
    Penalty(f64),
    Budget(f64),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DetectorCapabilities {
    pub changes: bool,
    pub penalty: bool,
    pub budget: bool,
}

impl DetectorCapabilities {
    pub const CHANGES_ONLY: Self = Self {
        changes: true,
        penalty: false,
        budget: false,
    };

    pub const PENALTY_ONLY: Self = Self {
        changes: false,
        penalty: true,
        budget: false,
    };

    pub fn supports(self, stop: Stop) -> bool {
        match stop {
            Stop::Changes(_) => self.changes,
            Stop::Penalty(_) => self.penalty,
            Stop::Budget(_) => self.budget,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Segmentation {
    pub breakpoints: Vec<usize>,
    pub segment_cost: f64,
    pub objective: f64,
}

impl Segmentation {
    pub fn new(
        breakpoints: Vec<usize>,
        segment_cost: f64,
        objective: f64,
        n_samples: usize,
        min_size: usize,
    ) -> Result<Self, Error> {
        validate_breakpoints(&breakpoints, n_samples, min_size)?;
        if !segment_cost.is_finite() {
            return Err(Error::NonFiniteObjective {
                value: segment_cost,
            });
        }
        if !objective.is_finite() {
            return Err(Error::NonFiniteObjective { value: objective });
        }
        Ok(Self {
            breakpoints,
            segment_cost,
            objective,
        })
    }
}

pub trait Detector<C: SegmentCost> {
    fn capabilities(&self) -> DetectorCapabilities;
    fn predict(&self, cost: &C, stop: Stop) -> Result<Segmentation, Error>;
}

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

pub fn validate_budget(budget: f64) -> Result<(), Error> {
    if !budget.is_finite() || budget < 0.0 {
        return Err(Error::InvalidBudget { value: budget });
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
    if objective_values_tied(candidate_cost, best_cost) {
        return candidate_breakpoints < best_breakpoints;
    }
    match candidate_cost.total_cmp(&best_cost) {
        Ordering::Less => true,
        Ordering::Equal | Ordering::Greater => false,
    }
}

#[inline]
pub fn objective_values_tied(left: f64, right: f64) -> bool {
    if left == right {
        return true;
    }
    if !left.is_finite() || !right.is_finite() {
        return false;
    }
    let scale = left.abs().max(right.abs());
    let tolerance = SCORE_ABSOLUTE_TOLERANCE + SCORE_RELATIVE_TOLERANCE * scale;
    (left - right).abs() <= tolerance
}

#[inline]
pub(crate) fn non_negative_score_tolerance(value: f64) -> f64 {
    debug_assert!(value.is_finite());
    debug_assert!(value >= 0.0);
    SCORE_ABSOLUTE_TOLERANCE + SCORE_RELATIVE_TOLERANCE * value
}

#[inline]
pub(crate) fn non_negative_candidate_is_significant(candidate: f64, incumbent: f64) -> bool {
    debug_assert!(candidate.is_finite());
    debug_assert!(incumbent.is_finite());
    debug_assert!(candidate >= 0.0);
    debug_assert!(candidate < incumbent);
    let tolerance = non_negative_score_tolerance(incumbent);
    incumbent - candidate > tolerance
}

#[inline]
pub(crate) fn non_negative_increase_is_significant(candidate: f64, incumbent: f64) -> bool {
    debug_assert!(incumbent.is_finite());
    debug_assert!(incumbent >= 0.0);
    debug_assert!(candidate > incumbent);
    if !candidate.is_finite() {
        return true;
    }
    let tolerance = SCORE_ABSOLUTE_TOLERANCE + SCORE_RELATIVE_TOLERANCE * candidate;
    candidate - incumbent > tolerance
}

#[cfg(test)]
#[path = "../../tests/unit/core/segmentation.rs"]
mod tests;
