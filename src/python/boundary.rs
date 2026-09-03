use std::any::Any;
use std::cell::Cell;
use std::panic::{self, catch_unwind, AssertUnwindSafe};
use std::sync::Once;

use pyo3::PyResult;

use super::error::RusturesError;

#[cfg(not(panic = "unwind"))]
compile_error!("the rustures Python extension requires panic=unwind");

thread_local! {
    static BOUNDARY_DEPTH: Cell<usize> = const { Cell::new(0) };
}

static INSTALL_PANIC_HOOK: Once = Once::new();

struct BoundaryGuard;

impl BoundaryGuard {
    fn enter() -> Self {
        BOUNDARY_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Self
    }
}

impl Drop for BoundaryGuard {
    fn drop(&mut self) {
        BOUNDARY_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

fn install_panic_hook() {
    INSTALL_PANIC_HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |information| {
            let handled_here = BOUNDARY_DEPTH
                .try_with(|depth| depth.get() > 0)
                .unwrap_or(false);
            if !handled_here {
                previous(information);
            }
        }));
    });
}

/// Converts an unwinding Rust panic into a regular Python exception.
///
/// PyO3 also protects its raw FFI trampolines, but its fallback is
/// `PanicException`, which derives directly from `BaseException`. Keeping our
/// own boundary inside every exported function makes internal failures
/// catchable with both `except RusturesError` and `except Exception`.
pub(super) fn catch_panic<T>(
    operation: &'static str,
    function: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    install_panic_hook();
    let result = {
        let _guard = BoundaryGuard::enter();
        catch_unwind(AssertUnwindSafe(function))
    };
    match result {
        Ok(result) => result,
        Err(payload) => Err(RusturesError::new_err(format!(
            "internal Rust panic while {operation}: {}",
            panic_message(payload.as_ref())
        ))),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "non-string panic payload"
    }
}

#[cfg(test)]
#[path = "../../tests/unit/python/boundary.rs"]
mod tests;
