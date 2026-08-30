mod boundary;
mod error;

use std::sync::Arc;

use numpy::{PyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

use self::boundary::catch_panic;
use self::error::RusturesError;
use crate::cost::CostL2 as RustCostL2;
use crate::kernel::{
    FusedKernel as RustFusedKernel, GammaPolicy as RustGammaPolicy,
    KernelBackend as RustKernelBackend, KernelCost as RustKernelCost, KernelKind as RustKernelKind,
};
use crate::search::{
    Binseg as RustBinseg, BottomUp as RustBottomUp, Dynp as RustDynp, Pelt as RustPelt,
    Window as RustWindow,
};
use crate::{
    datasets, metrics, validate_finite, validate_jump, validate_min_size, validate_signal_shape,
    Error, SegmentCost, SignalView, Stop,
};

#[pyfunction]
fn version() -> PyResult<&'static str> {
    catch_panic("getting the rustures version", || {
        Ok(env!("CARGO_PKG_VERSION"))
    })
}

#[pyfunction]
fn validate_signal(signal: PyReadonlyArrayDyn<'_, f64>) -> PyResult<(usize, usize)> {
    catch_panic("validating a signal", || {
        let shape = validate_signal_shape(signal.ndim(), signal.shape())?;
        validate_finite(signal.as_array().iter().copied(), shape)?;
        Ok((shape.n_samples, shape.n_features))
    })
}

#[pyfunction(name = "hausdorff")]
fn py_hausdorff(truth: Vec<usize>, prediction: Vec<usize>) -> PyResult<f64> {
    catch_panic("computing the Hausdorff metric", || {
        Ok(metrics::hausdorff(&truth, &prediction)?)
    })
}

#[pyfunction(name = "precision_recall", signature = (truth, prediction, margin = 10))]
fn py_precision_recall(
    truth: Vec<usize>,
    prediction: Vec<usize>,
    margin: usize,
) -> PyResult<(f64, f64)> {
    catch_panic("computing precision and recall", || {
        Ok(metrics::precision_recall(&truth, &prediction, margin)?)
    })
}

#[pyfunction(name = "rand_index")]
fn py_rand_index(truth: Vec<usize>, prediction: Vec<usize>) -> PyResult<f64> {
    catch_panic("computing the Rand index", || {
        Ok(metrics::rand_index(&truth, &prediction)?)
    })
}

fn dataset_array<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    n_samples: usize,
    n_features: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows: Vec<Vec<f64>> = values
        .chunks_exact(n_features)
        .map(<[f64]>::to_vec)
        .collect();
    debug_assert_eq!(rows.len(), n_samples);
    Ok(PyArray2::from_vec2(py, &rows)?)
}

macro_rules! dataset_binding {
    ($rust_name:ident, $python_name:literal, $generator:path) => {
        #[pyfunction(name = $python_name, signature = (n_samples, n_features = 1, n_bkps = 3, noise_std = 1.0, seed = 0))]
        fn $rust_name<'py>(
            py: Python<'py>,
            n_samples: usize,
            n_features: usize,
            n_bkps: usize,
            noise_std: f64,
            seed: u64,
        ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<usize>)> {
            catch_panic(concat!("generating ", $python_name), || {
                let (values, breakpoints) =
                    $generator(n_samples, n_features, n_bkps, noise_std, seed)?;
                Ok((dataset_array(py, values, n_samples, n_features)?, breakpoints))
            })
        }
    };
}

dataset_binding!(py_pw_constant, "pw_constant", datasets::piecewise_constant);
dataset_binding!(py_pw_linear, "pw_linear", datasets::piecewise_linear);
dataset_binding!(py_pw_normal, "pw_normal", datasets::piecewise_normal);
dataset_binding!(py_pw_wavy, "pw_wavy", datasets::piecewise_wavy);

fn cost_l2_from_numpy(signal: PyReadonlyArrayDyn<'_, f64>) -> Result<RustCostL2, Error> {
    let shape = validate_signal_shape(signal.ndim(), signal.shape())?;
    RustCostL2::from_values(signal.as_array().iter().copied(), shape)
}

/// Squared-error segment cost fitted from a one- or two-dimensional float64 NumPy array.
///
/// Fitting retains only owned prefix statistics; it does not retain the input array.
#[pyclass(name = "CostL2", module = "rustures._rustures")]
struct PyCostL2 {
    inner: Option<RustCostL2>,
}

impl PyCostL2 {
    fn fitted(&self) -> Result<&RustCostL2, Error> {
        self.inner
            .as_ref()
            .ok_or(Error::NotFitted { object: "CostL2" })
    }
}

/// Exact fixed-change dynamic programming detector.
#[pyclass(name = "Dynp", module = "rustures._rustures")]
struct PyDynp {
    model: String,
    detector: RustDynp,
    cost: Option<Arc<RustCostL2>>,
}

/// Exact penalized change-point detection using PELT pruning.
#[pyclass(name = "Pelt", module = "rustures._rustures")]
struct PyPelt {
    model: String,
    detector: RustPelt,
    cost: Option<Arc<RustCostL2>>,
}

#[pyclass(name = "Binseg", module = "rustures._rustures")]
struct PyBinseg {
    model: String,
    detector: RustBinseg,
    cost: Option<Arc<RustCostL2>>,
}

#[pyclass(name = "BottomUp", module = "rustures._rustures")]
struct PyBottomUp {
    model: String,
    detector: RustBottomUp,
    cost: Option<Arc<RustCostL2>>,
}

#[pyclass(name = "Window", module = "rustures._rustures")]
struct PyWindow {
    model: String,
    detector: RustWindow,
    cost: Option<Arc<RustCostL2>>,
}

#[pyclass(name = "KernelCPD", module = "rustures._rustures")]
struct PyKernelCPD {
    kernel_name: String,
    backend_name: String,
    kind: RustKernelKind,
    backend: PyKernelBackend,
    min_size: usize,
    jump: usize,
    max_gram_bytes: usize,
    cost: Option<Arc<RustKernelCost>>,
    fused: Option<Arc<RustFusedKernel>>,
}

#[derive(Clone, Copy)]
enum PyKernelBackend {
    Fused,
    Cost(RustKernelBackend),
}

fn l2_model(model: &str) -> Result<String, Error> {
    let model = model.to_ascii_lowercase();
    if model != "l2" {
        return Err(Error::UnsupportedModel { model });
    }
    Ok(model)
}

fn stopping_rule(
    n_bkps: Option<usize>,
    pen: Option<f64>,
    epsilon: Option<f64>,
) -> Result<Stop, Error> {
    match (n_bkps, pen, epsilon) {
        (Some(value), None, None) => Ok(Stop::Changes(value)),
        (None, Some(value), None) => Ok(Stop::Penalty(value)),
        (None, None, Some(value)) => Ok(Stop::Budget(value)),
        _ => Err(Error::InvalidStoppingRules),
    }
}

impl PyDynp {
    fn fitted(&self) -> Result<&Arc<RustCostL2>, Error> {
        self.cost
            .as_ref()
            .ok_or(Error::NotFitted { object: "Dynp" })
    }

    fn predict_inner(&self, py: Python<'_>, n_bkps: usize) -> PyResult<Vec<usize>> {
        let cost = Arc::clone(self.fitted()?);
        let detector = self.detector;
        let segmentation = py.detach(move || detector.predict_changes(cost.as_ref(), n_bkps))?;
        Ok(segmentation.breakpoints)
    }
}

impl PyPelt {
    fn fitted(&self) -> Result<&Arc<RustCostL2>, Error> {
        self.cost
            .as_ref()
            .ok_or(Error::NotFitted { object: "Pelt" })
    }

    fn predict_inner(&self, py: Python<'_>, pen: f64) -> PyResult<Vec<usize>> {
        let cost = Arc::clone(self.fitted()?);
        let detector = self.detector;
        let segmentation = py.detach(move || detector.predict_penalty(cost.as_ref(), pen))?;
        Ok(segmentation.breakpoints)
    }
}

macro_rules! fitted_cost {
    ($name:ident, $label:literal) => {
        impl $name {
            fn fitted(&self) -> Result<&Arc<RustCostL2>, Error> {
                self.cost
                    .as_ref()
                    .ok_or(Error::NotFitted { object: $label })
            }
        }
    };
}

fitted_cost!(PyBinseg, "Binseg");
fitted_cost!(PyBottomUp, "BottomUp");
fitted_cost!(PyWindow, "Window");

impl PyKernelCPD {
    fn ensure_fitted(&self) -> Result<(), Error> {
        if self.cost.is_none() && self.fused.is_none() {
            Err(Error::NotFitted {
                object: "KernelCPD",
            })
        } else {
            Ok(())
        }
    }
}

#[pymethods]
impl PyDynp {
    #[new]
    #[pyo3(signature = (model = "l2", min_size = 2, jump = 5))]
    fn new(model: &str, min_size: usize, jump: usize) -> PyResult<Self> {
        catch_panic("constructing Dynp", || {
            let model = model.to_ascii_lowercase();
            if model != "l2" {
                return Err(Error::UnsupportedModel { model }.into());
            }
            Ok(Self {
                model,
                detector: RustDynp::new(min_size, jump)?,
                cost: None,
            })
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting Dynp", || {
            slf.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            Ok(slf)
        })
    }

    fn predict(&self, py: Python<'_>, n_bkps: usize) -> PyResult<Vec<usize>> {
        catch_panic("predicting with Dynp", || self.predict_inner(py, n_bkps))
    }

    fn fit_predict(
        &mut self,
        py: Python<'_>,
        signal: PyReadonlyArrayDyn<'_, f64>,
        n_bkps: usize,
    ) -> PyResult<Vec<usize>> {
        catch_panic("fitting and predicting with Dynp", || {
            self.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            self.predict_inner(py, n_bkps)
        })
    }

    #[getter]
    fn model(&self) -> PyResult<&str> {
        catch_panic("reading Dynp.model", || Ok(self.model.as_str()))
    }

    #[getter]
    fn min_size(&self) -> PyResult<usize> {
        catch_panic("reading Dynp.min_size", || {
            Ok(self.detector.grid().min_size)
        })
    }

    #[getter]
    fn jump(&self) -> PyResult<usize> {
        catch_panic("reading Dynp.jump", || Ok(self.detector.grid().jump))
    }

    #[getter]
    fn is_fitted(&self) -> PyResult<bool> {
        catch_panic("reading Dynp.is_fitted", || Ok(self.cost.is_some()))
    }

    fn __repr__(&self) -> PyResult<String> {
        catch_panic("formatting Dynp", || {
            Ok(format!(
                "Dynp(model={:?}, min_size={}, jump={})",
                self.model,
                self.detector.grid().min_size,
                self.detector.grid().jump
            ))
        })
    }
}

#[pymethods]
impl PyPelt {
    #[new]
    #[pyo3(signature = (model = "l2", min_size = 2, jump = 5))]
    fn new(model: &str, min_size: usize, jump: usize) -> PyResult<Self> {
        catch_panic("constructing Pelt", || {
            let model = model.to_ascii_lowercase();
            if model != "l2" {
                return Err(Error::UnsupportedModel { model }.into());
            }
            Ok(Self {
                model,
                detector: RustPelt::new(min_size, jump)?,
                cost: None,
            })
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting Pelt", || {
            slf.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            Ok(slf)
        })
    }

    fn predict(&self, py: Python<'_>, pen: f64) -> PyResult<Vec<usize>> {
        catch_panic("predicting with Pelt", || self.predict_inner(py, pen))
    }

    fn fit_predict(
        &mut self,
        py: Python<'_>,
        signal: PyReadonlyArrayDyn<'_, f64>,
        pen: f64,
    ) -> PyResult<Vec<usize>> {
        catch_panic("fitting and predicting with Pelt", || {
            self.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            self.predict_inner(py, pen)
        })
    }

    #[getter]
    fn model(&self) -> PyResult<&str> {
        catch_panic("reading Pelt.model", || Ok(self.model.as_str()))
    }

    #[getter]
    fn min_size(&self) -> PyResult<usize> {
        catch_panic("reading Pelt.min_size", || {
            Ok(self.detector.grid().min_size)
        })
    }

    #[getter]
    fn jump(&self) -> PyResult<usize> {
        catch_panic("reading Pelt.jump", || Ok(self.detector.grid().jump))
    }

    #[getter]
    fn is_fitted(&self) -> PyResult<bool> {
        catch_panic("reading Pelt.is_fitted", || Ok(self.cost.is_some()))
    }

    fn __repr__(&self) -> PyResult<String> {
        catch_panic("formatting Pelt", || {
            Ok(format!(
                "Pelt(model={:?}, min_size={}, jump={})",
                self.model,
                self.detector.grid().min_size,
                self.detector.grid().jump
            ))
        })
    }
}

#[pymethods]
impl PyBinseg {
    #[new]
    #[pyo3(signature = (model = "l2", min_size = 2, jump = 5))]
    fn new(model: &str, min_size: usize, jump: usize) -> PyResult<Self> {
        catch_panic("constructing Binseg", || {
            Ok(Self {
                model: l2_model(model)?,
                detector: RustBinseg::new(min_size, jump)?,
                cost: None,
            })
        })
    }
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting Binseg", || {
            slf.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            Ok(slf)
        })
    }
    #[pyo3(signature = (n_bkps=None, pen=None, epsilon=None))]
    fn predict(
        &self,
        py: Python<'_>,
        n_bkps: Option<usize>,
        pen: Option<f64>,
        epsilon: Option<f64>,
    ) -> PyResult<Vec<usize>> {
        catch_panic("predicting with Binseg", || {
            let cost = Arc::clone(self.fitted()?);
            let detector = self.detector;
            let stop = stopping_rule(n_bkps, pen, epsilon)?;
            Ok(py
                .detach(move || detector.predict(cost.as_ref(), stop))?
                .breakpoints)
        })
    }
    #[pyo3(signature = (signal, n_bkps=None, pen=None, epsilon=None))]
    fn fit_predict(
        &mut self,
        py: Python<'_>,
        signal: PyReadonlyArrayDyn<'_, f64>,
        n_bkps: Option<usize>,
        pen: Option<f64>,
        epsilon: Option<f64>,
    ) -> PyResult<Vec<usize>> {
        catch_panic("fitting and predicting with Binseg", || {
            self.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            self.predict(py, n_bkps, pen, epsilon)
        })
    }
    #[getter]
    fn model(&self) -> PyResult<&str> {
        catch_panic("reading Binseg.model", || Ok(self.model.as_str()))
    }
    #[getter]
    fn min_size(&self) -> PyResult<usize> {
        catch_panic("reading Binseg.min_size", || {
            Ok(self.detector.grid().min_size)
        })
    }
    #[getter]
    fn jump(&self) -> PyResult<usize> {
        catch_panic("reading Binseg.jump", || Ok(self.detector.grid().jump))
    }
    #[getter]
    fn is_fitted(&self) -> PyResult<bool> {
        catch_panic("reading Binseg.is_fitted", || Ok(self.cost.is_some()))
    }
}

#[pymethods]
impl PyBottomUp {
    #[new]
    #[pyo3(signature = (model = "l2", min_size = 2, jump = 5))]
    fn new(model: &str, min_size: usize, jump: usize) -> PyResult<Self> {
        catch_panic("constructing BottomUp", || {
            Ok(Self {
                model: l2_model(model)?,
                detector: RustBottomUp::new(min_size, jump)?,
                cost: None,
            })
        })
    }
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting BottomUp", || {
            slf.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            Ok(slf)
        })
    }
    #[pyo3(signature = (n_bkps=None, pen=None, epsilon=None))]
    fn predict(
        &self,
        py: Python<'_>,
        n_bkps: Option<usize>,
        pen: Option<f64>,
        epsilon: Option<f64>,
    ) -> PyResult<Vec<usize>> {
        catch_panic("predicting with BottomUp", || {
            let cost = Arc::clone(self.fitted()?);
            let detector = self.detector;
            let stop = stopping_rule(n_bkps, pen, epsilon)?;
            Ok(py
                .detach(move || detector.predict(cost.as_ref(), stop))?
                .breakpoints)
        })
    }
    #[pyo3(signature = (signal, n_bkps=None, pen=None, epsilon=None))]
    fn fit_predict(
        &mut self,
        py: Python<'_>,
        signal: PyReadonlyArrayDyn<'_, f64>,
        n_bkps: Option<usize>,
        pen: Option<f64>,
        epsilon: Option<f64>,
    ) -> PyResult<Vec<usize>> {
        catch_panic("fitting and predicting with BottomUp", || {
            self.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            self.predict(py, n_bkps, pen, epsilon)
        })
    }
    #[getter]
    fn model(&self) -> PyResult<&str> {
        catch_panic("reading BottomUp.model", || Ok(self.model.as_str()))
    }
    #[getter]
    fn min_size(&self) -> PyResult<usize> {
        catch_panic("reading BottomUp.min_size", || {
            Ok(self.detector.grid().min_size)
        })
    }
    #[getter]
    fn jump(&self) -> PyResult<usize> {
        catch_panic("reading BottomUp.jump", || Ok(self.detector.grid().jump))
    }
    #[getter]
    fn is_fitted(&self) -> PyResult<bool> {
        catch_panic("reading BottomUp.is_fitted", || Ok(self.cost.is_some()))
    }
}

#[pymethods]
impl PyWindow {
    #[new]
    #[pyo3(signature = (width = 100, model = "l2", min_size = 2, jump = 5))]
    fn new(width: usize, model: &str, min_size: usize, jump: usize) -> PyResult<Self> {
        catch_panic("constructing Window", || {
            Ok(Self {
                model: l2_model(model)?,
                detector: RustWindow::new(width, min_size, jump)?,
                cost: None,
            })
        })
    }
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting Window", || {
            slf.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            Ok(slf)
        })
    }
    #[pyo3(signature = (n_bkps=None, pen=None, epsilon=None))]
    fn predict(
        &self,
        py: Python<'_>,
        n_bkps: Option<usize>,
        pen: Option<f64>,
        epsilon: Option<f64>,
    ) -> PyResult<Vec<usize>> {
        catch_panic("predicting with Window", || {
            let cost = Arc::clone(self.fitted()?);
            let detector = self.detector;
            let stop = stopping_rule(n_bkps, pen, epsilon)?;
            Ok(py
                .detach(move || detector.predict(cost.as_ref(), stop))?
                .breakpoints)
        })
    }
    #[pyo3(signature = (signal, n_bkps=None, pen=None, epsilon=None))]
    fn fit_predict(
        &mut self,
        py: Python<'_>,
        signal: PyReadonlyArrayDyn<'_, f64>,
        n_bkps: Option<usize>,
        pen: Option<f64>,
        epsilon: Option<f64>,
    ) -> PyResult<Vec<usize>> {
        catch_panic("fitting and predicting with Window", || {
            self.cost = Some(Arc::new(cost_l2_from_numpy(signal)?));
            self.predict(py, n_bkps, pen, epsilon)
        })
    }
    #[getter]
    fn model(&self) -> PyResult<&str> {
        catch_panic("reading Window.model", || Ok(self.model.as_str()))
    }
    #[getter]
    fn width(&self) -> PyResult<usize> {
        catch_panic("reading Window.width", || Ok(self.detector.width()))
    }
    #[getter]
    fn min_size(&self) -> PyResult<usize> {
        catch_panic("reading Window.min_size", || {
            Ok(self.detector.grid().min_size)
        })
    }
    #[getter]
    fn jump(&self) -> PyResult<usize> {
        catch_panic("reading Window.jump", || Ok(self.detector.grid().jump))
    }
    #[getter]
    fn is_fitted(&self) -> PyResult<bool> {
        catch_panic("reading Window.is_fitted", || Ok(self.cost.is_some()))
    }
}

#[pymethods]
impl PyKernelCPD {
    #[new]
    #[pyo3(signature = (kernel = "rbf", min_size = 2, jump = 1, gamma = None, gamma_policy = "exact", gamma_samples = 10_000, seed = 0, backend = "fused", max_gram_bytes = 536_870_912))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kernel: &str,
        min_size: usize,
        jump: usize,
        gamma: Option<f64>,
        gamma_policy: &str,
        gamma_samples: usize,
        seed: u64,
        backend: &str,
        max_gram_bytes: usize,
    ) -> PyResult<Self> {
        catch_panic("constructing KernelCPD", || {
            validate_min_size(min_size)?;
            validate_jump(jump)?;
            let kernel_name = kernel.to_ascii_lowercase();
            let kind = match kernel_name.as_str() {
                "linear" => RustKernelKind::Linear,
                "cosine" => RustKernelKind::Cosine,
                "rbf" => {
                    let policy = if let Some(value) = gamma {
                        RustGammaPolicy::Fixed(value)
                    } else {
                        match gamma_policy.to_ascii_lowercase().as_str() {
                            "exact" => RustGammaPolicy::ExactMedian,
                            "sampled" => RustGammaPolicy::SampledMedian {
                                pairs: gamma_samples,
                                seed,
                            },
                            _ => {
                                return Err(Error::UnsupportedGammaPolicy {
                                    policy: gamma_policy.to_string(),
                                }
                                .into());
                            }
                        }
                    };
                    RustKernelKind::Rbf(policy)
                }
                _ => {
                    return Err(Error::UnsupportedKernel {
                        kernel: kernel.to_string(),
                    }
                    .into());
                }
            };
            let backend_name = backend.to_ascii_lowercase();
            let backend = match backend_name.as_str() {
                "fused" => PyKernelBackend::Fused,
                "full" => PyKernelBackend::Cost(RustKernelBackend::FullGram),
                "streaming" => PyKernelBackend::Cost(RustKernelBackend::Streaming),
                _ => {
                    return Err(Error::UnsupportedKernelBackend {
                        backend: backend.to_string(),
                    }
                    .into());
                }
            };
            Ok(Self {
                kernel_name,
                backend_name,
                kind,
                backend,
                min_size,
                jump,
                max_gram_bytes,
                cost: None,
                fused: None,
            })
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting KernelCPD", || {
            let shape = validate_signal_shape(signal.ndim(), signal.shape())?;
            let values: Vec<f64> = signal.as_array().iter().copied().collect();
            validate_finite(values.iter().copied(), shape)?;
            let kind = slf.kind;
            let backend = slf.backend;
            let limit = slf.max_gram_bytes;
            let min_size = slf.min_size;
            let jump = slf.jump;
            let (cost, fused) = py.detach(move || -> Result<_, Error> {
                let view = SignalView::new(&values, shape.n_samples, shape.n_features)?;
                match backend {
                    PyKernelBackend::Fused => Ok((
                        None,
                        Some(Arc::new(RustFusedKernel::fit(view, kind, min_size, jump)?)),
                    )),
                    PyKernelBackend::Cost(backend) => Ok((
                        Some(Arc::new(RustKernelCost::fit(view, kind, backend, limit)?)),
                        None,
                    )),
                }
            })?;
            slf.cost = cost;
            slf.fused = fused;
            Ok(slf)
        })
    }

    #[pyo3(signature = (n_bkps=None, pen=None))]
    fn predict(
        &self,
        py: Python<'_>,
        n_bkps: Option<usize>,
        pen: Option<f64>,
    ) -> PyResult<Vec<usize>> {
        catch_panic("predicting with KernelCPD", || {
            if usize::from(n_bkps.is_some()) + usize::from(pen.is_some()) != 1 {
                return Err(Error::InvalidStoppingRules.into());
            }
            self.ensure_fitted()?;
            let fused = self.fused.as_ref().map(Arc::clone);
            let cost = self.cost.as_ref().map(Arc::clone);
            let min_size = self.min_size;
            let jump = self.jump;
            Ok(py
                .detach(move || {
                    if let Some(detector) = fused {
                        match (n_bkps, pen) {
                            (Some(changes), None) => detector.predict_changes(changes),
                            (None, Some(penalty)) => detector.predict_penalty(penalty),
                            _ => Err(Error::InvalidStoppingRules),
                        }
                    } else {
                        let cost = cost.ok_or(Error::NotFitted {
                            object: "KernelCPD",
                        })?;
                        match (n_bkps, pen) {
                            (Some(changes), None) => RustDynp::new(min_size, jump)?
                                .predict_changes(cost.as_ref(), changes),
                            (None, Some(penalty)) => RustPelt::new(min_size, jump)?
                                .predict_penalty(cost.as_ref(), penalty),
                            _ => Err(Error::InvalidStoppingRules),
                        }
                    }
                })?
                .breakpoints)
        })
    }

    #[pyo3(signature = (signal, n_bkps=None, pen=None))]
    fn fit_predict(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        signal: PyReadonlyArrayDyn<'_, f64>,
        n_bkps: Option<usize>,
        pen: Option<f64>,
    ) -> PyResult<Vec<usize>> {
        catch_panic("fitting and predicting with KernelCPD", || {
            let shape = validate_signal_shape(signal.ndim(), signal.shape())?;
            let values: Vec<f64> = signal.as_array().iter().copied().collect();
            validate_finite(values.iter().copied(), shape)?;
            let kind = slf.kind;
            let backend = slf.backend;
            let limit = slf.max_gram_bytes;
            let min_size = slf.min_size;
            let jump = slf.jump;
            let (cost, fused) = py.detach(move || -> Result<_, Error> {
                let view = SignalView::new(&values, shape.n_samples, shape.n_features)?;
                match backend {
                    PyKernelBackend::Fused => Ok((
                        None,
                        Some(Arc::new(RustFusedKernel::fit(view, kind, min_size, jump)?)),
                    )),
                    PyKernelBackend::Cost(backend) => Ok((
                        Some(Arc::new(RustKernelCost::fit(view, kind, backend, limit)?)),
                        None,
                    )),
                }
            })?;
            slf.cost = cost;
            slf.fused = fused;
            slf.predict(py, n_bkps, pen)
        })
    }

    #[getter]
    fn kernel(&self) -> PyResult<&str> {
        catch_panic("reading KernelCPD.kernel", || Ok(self.kernel_name.as_str()))
    }
    #[getter]
    fn backend(&self) -> PyResult<&str> {
        catch_panic("reading KernelCPD.backend", || {
            Ok(self.backend_name.as_str())
        })
    }
    #[getter]
    fn min_size(&self) -> PyResult<usize> {
        catch_panic("reading KernelCPD.min_size", || Ok(self.min_size))
    }
    #[getter]
    fn jump(&self) -> PyResult<usize> {
        catch_panic("reading KernelCPD.jump", || Ok(self.jump))
    }
    #[getter]
    fn gamma(&self) -> PyResult<Option<f64>> {
        catch_panic("reading KernelCPD.gamma", || {
            Ok(self
                .fused
                .as_ref()
                .and_then(|detector| detector.gamma())
                .or_else(|| self.cost.as_ref().and_then(|cost| cost.gamma())))
        })
    }
    #[getter]
    fn stored_gram_entries(&self) -> PyResult<usize> {
        catch_panic("reading KernelCPD.stored_gram_entries", || {
            self.ensure_fitted()?;
            Ok(self
                .cost
                .as_ref()
                .map_or(0, |cost| cost.stored_gram_entries()))
        })
    }
    #[getter]
    fn is_fitted(&self) -> PyResult<bool> {
        catch_panic("reading KernelCPD.is_fitted", || {
            Ok(self.cost.is_some() || self.fused.is_some())
        })
    }
}

#[pymethods]
impl PyCostL2 {
    #[new]
    fn new() -> PyResult<Self> {
        catch_panic("constructing CostL2", || Ok(Self { inner: None }))
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting CostL2", || {
            slf.inner = Some(cost_l2_from_numpy(signal)?);
            Ok(slf)
        })
    }

    fn error(&self, start: usize, end: usize) -> PyResult<f64> {
        catch_panic("computing a CostL2 segment", || {
            Ok(self.fitted()?.cost(start..end)?)
        })
    }

    fn sum_of_costs(&self, breakpoints: Vec<usize>) -> PyResult<f64> {
        catch_panic("summing CostL2 segments", || {
            Ok(self.fitted()?.sum_of_costs(&breakpoints)?)
        })
    }

    #[getter]
    fn n_samples(&self) -> PyResult<usize> {
        catch_panic("reading CostL2.n_samples", || {
            Ok(self.fitted()?.n_samples())
        })
    }

    #[getter]
    fn n_features(&self) -> PyResult<usize> {
        catch_panic("reading CostL2.n_features", || {
            Ok(self.fitted()?.n_features())
        })
    }

    #[getter]
    fn min_size(&self) -> PyResult<usize> {
        catch_panic("reading CostL2.min_size", || Ok(1))
    }

    #[getter]
    fn is_fitted(&self) -> PyResult<bool> {
        catch_panic("reading CostL2.is_fitted", || Ok(self.inner.is_some()))
    }

    fn __repr__(&self) -> PyResult<String> {
        catch_panic("formatting CostL2", || {
            Ok(match &self.inner {
                Some(cost) => format!(
                    "CostL2(n_samples={}, n_features={})",
                    cost.n_samples(),
                    cost.n_features()
                ),
                None => "CostL2()".to_owned(),
            })
        })
    }
}

#[cfg(feature = "panic-test-hook")]
#[pyfunction]
fn _panic_test_hook() -> PyResult<()> {
    catch_panic("running the panic test hook", || {
        panic!("intentional panic-test-hook panic")
    })
}

#[pymodule]
fn _rustures(module: &Bound<'_, PyModule>) -> PyResult<()> {
    catch_panic("initializing the rustures extension", || {
        module.add("RusturesError", module.py().get_type::<RusturesError>())?;
        module.add("__version__", env!("CARGO_PKG_VERSION"))?;
        module.add_class::<PyCostL2>()?;
        module.add_class::<PyDynp>()?;
        module.add_class::<PyPelt>()?;
        module.add_class::<PyBinseg>()?;
        module.add_class::<PyBottomUp>()?;
        module.add_class::<PyWindow>()?;
        module.add_class::<PyKernelCPD>()?;
        module.add_function(wrap_pyfunction!(version, module)?)?;
        module.add_function(wrap_pyfunction!(validate_signal, module)?)?;
        module.add_function(wrap_pyfunction!(py_hausdorff, module)?)?;
        module.add_function(wrap_pyfunction!(py_precision_recall, module)?)?;
        module.add_function(wrap_pyfunction!(py_rand_index, module)?)?;
        module.add_function(wrap_pyfunction!(py_pw_constant, module)?)?;
        module.add_function(wrap_pyfunction!(py_pw_linear, module)?)?;
        module.add_function(wrap_pyfunction!(py_pw_normal, module)?)?;
        module.add_function(wrap_pyfunction!(py_pw_wavy, module)?)?;
        #[cfg(feature = "panic-test-hook")]
        module.add_function(wrap_pyfunction!(_panic_test_hook, module)?)?;
        Ok(())
    })
}
