use pyo3::prelude::*;

use super::boundary::catch_panic;

#[pyfunction]
fn _panic_test_hook() -> PyResult<()> {
    catch_panic("running the panic test hook", || {
        panic!("intentional panic-test-hook panic")
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_panic_test_hook, module)?)?;
    Ok(())
}
