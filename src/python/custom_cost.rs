use std::ops::Range;
use std::sync::Mutex;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::{validate_segment, Error, SegmentCost};

const CALLBACK_CONTEXT: &str = "executing a Python custom cost callback";

#[derive(Default)]
struct BatchCache {
    end: Option<usize>,
    starts: Vec<usize>,
    values: Vec<f64>,
}

/// Binding-only adapter from the Python custom-cost protocol to `SegmentCost`.
///
/// The Rust algorithms still see the ordinary `SegmentCost` contract. Python
/// exceptions are parked in `callback_error` while the trait reports a sentinel
/// Rust error; the binding retrieves and returns the original `PyErr`, including
/// its traceback.
pub(super) struct PythonSegmentCost {
    object: Py<PyAny>,
    n_samples: usize,
    n_features: usize,
    min_size: usize,
    jump: usize,
    has_error_many: bool,
    batch_cache: Mutex<BatchCache>,
    callback_error: Mutex<Option<PyErr>>,
}

impl PythonSegmentCost {
    pub(super) fn protocol_min_size(py: Python<'_>, object: &Py<PyAny>) -> PyResult<usize> {
        let bound = object.bind(py);
        for method in ["fit", "error"] {
            let attribute = bound.getattr(method).map_err(|_| {
                PyTypeError::new_err(format!(
                    "custom_cost must define a callable {method}() method"
                ))
            })?;
            if !attribute.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "custom_cost.{method} must be callable"
                )));
            }
        }

        let value = bound.getattr("min_size").map_err(|_| {
            PyTypeError::new_err("custom_cost must define a positive integer min_size attribute")
        })?;
        let value = value
            .extract::<isize>()
            .map_err(|_| PyTypeError::new_err("custom_cost.min_size must be a positive integer"))?;
        if value < 1 {
            return Err(PyValueError::new_err(format!(
                "custom_cost.min_size must be positive, got {value}"
            )));
        }
        Ok(value as usize)
    }

    pub(super) fn fit(
        py: Python<'_>,
        object: &Py<PyAny>,
        signal: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        object.bind(py).call_method1("fit", (signal,))?;
        Ok(())
    }

    pub(super) fn new(
        py: Python<'_>,
        object: Py<PyAny>,
        n_samples: usize,
        n_features: usize,
        min_size: usize,
        jump: usize,
    ) -> PyResult<Self> {
        let has_error_many = if object.bind(py).hasattr("error_many")? {
            let method = object.bind(py).getattr("error_many")?;
            if !method.is_callable() {
                return Err(PyTypeError::new_err(
                    "custom_cost.error_many must be callable when provided",
                ));
            }
            true
        } else {
            false
        };

        Ok(Self {
            object,
            n_samples,
            n_features,
            min_size,
            jump,
            has_error_many,
            batch_cache: Mutex::new(BatchCache::default()),
            callback_error: Mutex::new(None),
        })
    }

    pub(super) fn uses_batch_callback(&self) -> bool {
        self.has_error_many
    }

    pub(super) fn take_callback_error(&self) -> Option<PyErr> {
        self.callback_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }

    fn record_callback_error(&self, error: PyErr) -> Error {
        *self
            .callback_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error);
        Error::NumericalFailure {
            context: CALLBACK_CONTEXT,
        }
    }

    fn scalar_cost(&self, segment: Range<usize>) -> PyResult<f64> {
        Python::attach(|py| {
            self.object
                .bind(py)
                .call_method1("error", (segment.start, segment.end))?
                .extract::<f64>()
        })
    }

    fn admissible_starts(&self, requested_start: usize, end: usize) -> Vec<usize> {
        let mut starts = Vec::new();
        if end >= self.min_size {
            starts.push(0);
        }

        let mut start = self.jump;
        while start < end {
            if end - start >= self.min_size {
                starts.push(start);
            }
            match start.checked_add(self.jump) {
                Some(next) => start = next,
                None => break,
            }
        }

        if starts.binary_search(&requested_start).is_err() {
            starts.push(requested_start);
            starts.sort_unstable();
            starts.dedup();
        }
        starts
    }

    fn batch_costs(&self, starts: &[usize], end: usize) -> PyResult<Vec<f64>> {
        Python::attach(|py| {
            let starts_array = PyArray1::from_vec(py, starts.to_vec());
            let ends_array = PyArray1::from_vec(py, vec![end; starts.len()]);
            let result = self
                .object
                .bind(py)
                .call_method1("error_many", (starts_array.as_any(), ends_array.as_any()))?;
            let result = result.extract::<PyReadonlyArray1<'_, f64>>().map_err(|_| {
                PyTypeError::new_err(
                    "custom_cost.error_many must return a one-dimensional float64 NumPy array",
                )
            })?;
            let values: Vec<f64> = result.as_array().iter().copied().collect();
            if values.len() != starts.len() {
                return Err(PyValueError::new_err(format!(
                    "custom_cost.error_many returned {} values for {} segments",
                    values.len(),
                    starts.len()
                )));
            }
            if let Some((index, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(PyValueError::new_err(format!(
                    "custom_cost.error_many returned a non-finite value {value} for segment [{}, {end})",
                    starts[index]
                )));
            }
            Ok(values)
        })
    }

    fn batch_cost(&self, segment: Range<usize>) -> PyResult<f64> {
        {
            let cache = self
                .batch_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cache.end == Some(segment.end) {
                if let Ok(index) = cache.starts.binary_search(&segment.start) {
                    return Ok(cache.values[index]);
                }
            }
        }

        let starts = self.admissible_starts(segment.start, segment.end);
        let values = self.batch_costs(&starts, segment.end)?;
        let index = starts.binary_search(&segment.start).map_err(|_| {
            PyValueError::new_err("internal custom-cost batch omitted the requested segment")
        })?;
        let value = values[index];

        let mut cache = self
            .batch_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.end = Some(segment.end);
        cache.starts = starts;
        cache.values = values;
        Ok(value)
    }
}

impl SegmentCost for PythonSegmentCost {
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn min_size(&self) -> usize {
        self.min_size
    }

    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        validate_segment(segment.clone(), self.n_samples, self.min_size)?;
        let value = if self.has_error_many {
            self.batch_cost(segment)
        } else {
            self.scalar_cost(segment)
        }
        .map_err(|error| self.record_callback_error(error))?;

        if !value.is_finite() {
            return Err(self.record_callback_error(PyValueError::new_err(format!(
                "custom cost for segment returned a non-finite value: {value}"
            ))));
        }
        Ok(value)
    }

    // An arbitrary Python cost has not proved the PELT pruning inequality.
    // Returning None forces the exact unpruned optimal-partitioning path.
    fn pelt_pruning_constant(&self) -> Option<f64> {
        None
    }
}
