mod error;
mod segmentation;
mod signal;

#[cfg(test)]
mod oracle;

pub use error::Error;
pub use segmentation::{
    candidate_is_better, partition_cost, validate_breakpoints, validate_jump, validate_min_size,
    validate_penalty, validate_segment,
};
pub use signal::{validate_finite, validate_signal_shape, SignalShape};

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_rustures, RusturesError, PyException);

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn validate_signal(signal: PyReadonlyArrayDyn<'_, f64>) -> PyResult<(usize, usize)> {
    let shape = validate_signal_shape(signal.ndim(), signal.shape())?;
    validate_finite(signal.as_array().iter().copied(), shape)?;
    Ok((shape.n_samples, shape.n_features))
}

#[pymodule]
fn _rustures(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("RusturesError", module.py().get_type::<RusturesError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(validate_signal, module)?)?;
    Ok(())
}
