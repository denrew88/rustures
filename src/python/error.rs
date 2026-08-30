use pyo3::create_exception;
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

use crate::Error;

create_exception!(_rustures, RusturesError, PyRuntimeError);

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        let message = value.to_string();
        match value {
            Error::NotFitted { .. } => PyRuntimeError::new_err(message),
            Error::GramMemoryLimit { .. } => PyMemoryError::new_err(message),
            Error::NumericalFailure { .. } | Error::NonFiniteObjective { .. } => {
                RusturesError::new_err(message)
            }
            _ => PyValueError::new_err(message),
        }
    }
}
