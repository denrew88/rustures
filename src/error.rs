use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    #[error("signal must contain at least one sample and one feature")]
    EmptySignal,

    #[error("signal must be one- or two-dimensional, got {ndim} dimensions")]
    InvalidSignalDimension { ndim: usize },

    #[error("signal contains a non-finite value at row {row}, column {column}")]
    NonFiniteInput { row: usize, column: usize },

    #[error("invalid segment [{start}, {end}) for a signal of length {n_samples}")]
    InvalidRange {
        start: usize,
        end: usize,
        n_samples: usize,
    },

    #[error("segment [{start}, {end}) has length {length}, below minimum {minimum}")]
    SegmentTooShort {
        start: usize,
        end: usize,
        length: usize,
        minimum: usize,
    },

    #[error("min_size must be positive, got {value}")]
    InvalidMinSize { value: usize },

    #[error("jump must be positive, got {value}")]
    InvalidJump { value: usize },

    #[error("penalty must be finite and positive, got {value}")]
    InvalidPenalty { value: f64 },

    #[error("breakpoints must not be empty")]
    EmptyBreakpoints,

    #[error("breakpoint at position {position} is {value}, but breakpoints must be strictly increasing and no larger than {n_samples}")]
    InvalidBreakpoint {
        position: usize,
        value: usize,
        n_samples: usize,
    },

    #[error("last breakpoint must equal n_samples ({n_samples}), got {actual}")]
    MissingTerminalBreakpoint { actual: usize, n_samples: usize },

    #[error("exhaustive test oracle supports at most {maximum} samples, got {actual}")]
    ExhaustiveLimitExceeded { actual: usize, maximum: usize },

    #[error("objective value must be finite, got {value}")]
    NonFiniteObjective { value: f64 },
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        PyValueError::new_err(value.to_string())
    }
}
