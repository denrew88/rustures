use std::ops::Range;
use std::sync::Mutex;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::{validate_segment, Error, SegmentCost};

const CALLBACK_CONTEXT: &str = "executing a Python custom cost callback";

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
    has_error_many: bool,
    pelt_pruning_constant: Option<f64>,
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
        let pelt_pruning_constant = if object.bind(py).hasattr("pelt_pruning_constant")? {
            let value = object.bind(py).getattr("pelt_pruning_constant")?;
            if value.is_none() {
                None
            } else {
                let value = value.extract::<f64>().map_err(|_| {
                    PyTypeError::new_err(
                        "custom_cost.pelt_pruning_constant must be a finite float or None",
                    )
                })?;
                if !value.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "custom_cost.pelt_pruning_constant must be finite, got {value}"
                    )));
                }
                Some(value)
            }
        } else {
            None
        };

        Ok(Self {
            object,
            n_samples,
            n_features,
            min_size,
            has_error_many,
            pelt_pruning_constant,
            callback_error: Mutex::new(None),
        })
    }

    pub(super) fn uses_batch_callback(&self) -> bool {
        self.has_error_many
    }

    pub(super) fn uses_pelt_pruning(&self) -> bool {
        self.pelt_pruning_constant.is_some()
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

    fn batch_costs_into(
        &self,
        starts: &[usize],
        end: usize,
        output: &mut Vec<f64>,
    ) -> PyResult<()> {
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
            let values = result.as_array();
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
            output.clear();
            output.extend(values.iter().copied());
            Ok(())
        })
    }

    fn batch_cost(&self, segment: Range<usize>) -> PyResult<f64> {
        // `costs_ending_at` performs the useful endpoint-wide batching used by
        // Dynp and Pelt. A standalone `cost` request needs exactly one segment;
        // expanding it to every grid-aligned start wastes work during final
        // objective reconstruction and in algorithms with irregular queries.
        let starts = [segment.start];
        let mut values = Vec::with_capacity(1);
        self.batch_costs_into(&starts, segment.end, &mut values)?;
        values.pop().ok_or_else(|| {
            PyValueError::new_err("custom_cost.error_many omitted the requested segment")
        })
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

    fn costs_ending_at(
        &self,
        starts: &[usize],
        end: usize,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        output.clear();
        output
            .try_reserve(starts.len())
            .map_err(|_| Error::AllocationFailure {
                context: "allocating a Python custom-cost endpoint batch",
            })?;
        for &start in starts {
            validate_segment(start..end, self.n_samples, self.min_size)?;
        }
        if starts.is_empty() {
            return Ok(());
        }

        if self.has_error_many {
            return self
                .batch_costs_into(starts, end, output)
                .map_err(|error| self.record_callback_error(error));
        }

        for &start in starts {
            let value = self
                .scalar_cost(start..end)
                .map_err(|error| self.record_callback_error(error))?;
            if !value.is_finite() {
                return Err(self.record_callback_error(PyValueError::new_err(format!(
                    "custom cost for segment returned a non-finite value: {value}"
                ))));
            }
            output.push(value);
        }
        Ok(())
    }

    fn pelt_pruning_constant(&self) -> Option<f64> {
        self.pelt_pruning_constant
    }
}
