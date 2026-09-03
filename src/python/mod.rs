mod boundary;
mod custom_cost;
mod error;
#[cfg(feature = "panic-test-hook")]
mod test_support;

use std::sync::Arc;

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use self::boundary::catch_panic;
use self::custom_cost::PythonSegmentCost;
use self::error::RusturesError;
use crate::cost::{CostL2 as RustCostL2, CostModel as RustCostModel, CostSpec as RustCostSpec};
use crate::kernel::{
    FusedKernel as RustFusedKernel, GammaPolicy as RustGammaPolicy,
    KernelBackend as RustKernelBackend, KernelCost as RustKernelCost, KernelKind as RustKernelKind,
};
use crate::search::{
    Binseg as RustBinseg, BottomUp as RustBottomUp, Dynp as RustDynp, L1Potts as RustL1Potts,
    Pelt as RustPelt, Window as RustWindow,
};
use crate::{
    datasets, metrics, validate_breakpoints, validate_finite, validate_jump, validate_min_size,
    validate_signal_shape, Error, SegmentCost, SignalView, Stop,
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
    let expected = n_samples
        .checked_mul(n_features)
        .ok_or_else(|| PyValueError::new_err("synthetic dataset dimensions overflow"))?;
    if values.len() != expected {
        return Err(PyValueError::new_err(format!(
            "synthetic dataset produced {} values, expected {expected}",
            values.len()
        )));
    }

    // `from_vec` transfers the single contiguous Rust allocation to NumPy.
    // Reshape only changes metadata, avoiding n_samples row allocations and a
    // second full copy through `Vec<Vec<f64>>`.
    PyArray1::from_vec(py, values).reshape([n_samples, n_features])
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

fn cost_model_from_numpy(
    signal: PyReadonlyArrayDyn<'_, f64>,
    spec: &RustCostSpec,
) -> Result<RustCostModel, Error> {
    let shape = validate_signal_shape(signal.ndim(), signal.shape())?;
    let values: Vec<f64> = signal.as_array().iter().copied().collect();
    let view = SignalView::new(&values, shape.n_samples, shape.n_features)?;
    spec.fit(view)
}

fn custom_cost_from_numpy(
    py: Python<'_>,
    signal: &PyReadonlyArrayDyn<'_, f64>,
    source: &Py<PyAny>,
    effective_min_size: usize,
    detector: &'static str,
) -> PyResult<Arc<PythonSegmentCost>> {
    let shape = validate_signal_shape(signal.ndim(), signal.shape())?;
    validate_finite(signal.as_array().iter().copied(), shape)?;
    PythonSegmentCost::fit(py, source, signal.as_any())?;
    let fitted_min_size = PythonSegmentCost::protocol_min_size(py, source)?;
    if fitted_min_size > effective_min_size {
        return Err(PyValueError::new_err(format!(
            "custom_cost.min_size changed to {fitted_min_size} during fit; construct {detector} with min_size at least {fitted_min_size}"
        )));
    }
    Ok(Arc::new(PythonSegmentCost::new(
        py,
        source.clone_ref(py),
        shape.n_samples,
        shape.n_features,
        effective_min_size,
    )?))
}

fn l1_potts_from_numpy(
    signal: PyReadonlyArrayDyn<'_, f64>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
) -> Result<RustL1Potts, Error> {
    let shape = validate_signal_shape(signal.ndim(), signal.shape())?;
    let values: Vec<f64> = signal.as_array().iter().copied().collect();
    let weights = weights.map(|array| array.as_array().iter().copied().collect::<Vec<_>>());
    let view = SignalView::new(&values, shape.n_samples, shape.n_features)?;
    RustL1Potts::fit(view, weights.as_deref())
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

fn fitted_cost_model<'a>(
    inner: &'a Option<RustCostModel>,
    object: &'static str,
) -> Result<&'a RustCostModel, Error> {
    inner.as_ref().ok_or(Error::NotFitted { object })
}

fn sum_model_costs(cost: &RustCostModel, breakpoints: &[usize]) -> Result<f64, Error> {
    validate_breakpoints(breakpoints, cost.n_samples(), cost.min_size())?;
    let mut start = 0;
    let mut total = 0.0;
    let mut correction = 0.0;
    for &end in breakpoints {
        let value = cost.cost(start..end)?;
        let adjusted = value - correction;
        let next = total + adjusted;
        correction = (next - total) - adjusted;
        total = next;
        start = end;
    }
    if !total.is_finite() {
        return Err(Error::NonFiniteObjective { value: total });
    }
    Ok(total)
}

macro_rules! simple_cost_binding {
    ($rust_name:ident, $python_name:literal, $spec:expr) => {
        #[pyclass(name = $python_name, module = "rustures._rustures")]
        struct $rust_name {
            inner: Option<RustCostModel>,
        }

        #[pymethods]
        impl $rust_name {
            #[new]
            fn new() -> PyResult<Self> {
                catch_panic(concat!("constructing ", $python_name), || {
                    Ok(Self { inner: None })
                })
            }

            fn fit<'py>(
                mut slf: PyRefMut<'py, Self>,
                signal: PyReadonlyArrayDyn<'py, f64>,
            ) -> PyResult<PyRefMut<'py, Self>> {
                catch_panic(concat!("fitting ", $python_name), || {
                    slf.inner = Some(cost_model_from_numpy(signal, &$spec)?);
                    Ok(slf)
                })
            }

            fn error(&self, start: usize, end: usize) -> PyResult<f64> {
                catch_panic(concat!("computing a ", $python_name, " segment"), || {
                    Ok(fitted_cost_model(&self.inner, $python_name)?.cost(start..end)?)
                })
            }

            fn sum_of_costs(&self, breakpoints: Vec<usize>) -> PyResult<f64> {
                catch_panic(concat!("summing ", $python_name, " segments"), || {
                    Ok(sum_model_costs(
                        fitted_cost_model(&self.inner, $python_name)?,
                        &breakpoints,
                    )?)
                })
            }

            #[getter]
            fn n_samples(&self) -> PyResult<usize> {
                catch_panic(concat!("reading ", $python_name, ".n_samples"), || {
                    Ok(fitted_cost_model(&self.inner, $python_name)?.n_samples())
                })
            }

            #[getter]
            fn n_features(&self) -> PyResult<usize> {
                catch_panic(concat!("reading ", $python_name, ".n_features"), || {
                    Ok(fitted_cost_model(&self.inner, $python_name)?.n_features())
                })
            }

            #[getter]
            fn min_size(&self) -> PyResult<usize> {
                catch_panic(concat!("reading ", $python_name, ".min_size"), || {
                    Ok($spec.minimum_size_hint())
                })
            }

            #[getter]
            fn is_fitted(&self) -> PyResult<bool> {
                catch_panic(concat!("reading ", $python_name, ".is_fitted"), || {
                    Ok(self.inner.is_some())
                })
            }

            fn __repr__(&self) -> PyResult<String> {
                catch_panic(concat!("formatting ", $python_name), || {
                    Ok(if let Some(cost) = &self.inner {
                        format!(
                            concat!($python_name, "(n_samples={}, n_features={})"),
                            cost.n_samples(),
                            cost.n_features()
                        )
                    } else {
                        concat!($python_name, "()").to_owned()
                    })
                })
            }
        }
    };
}

simple_cost_binding!(PyCostL1, "CostL1", RustCostSpec::L1);
simple_cost_binding!(PyCostRank, "CostRank", RustCostSpec::Rank);
simple_cost_binding!(PyCostLinear, "CostLinear", RustCostSpec::Linear);
simple_cost_binding!(PyCostCLinear, "CostCLinear", RustCostSpec::CLinear);

macro_rules! configured_cost_methods {
    ($rust_name:ident, $python_name:literal) => {
        impl $rust_name {
            fn fitted(&self) -> Result<&RustCostModel, Error> {
                fitted_cost_model(&self.inner, $python_name)
            }
        }
    };
}

#[pyclass(name = "CostNormal", module = "rustures._rustures")]
struct PyCostNormal {
    ridge: f64,
    inner: Option<RustCostModel>,
}
configured_cost_methods!(PyCostNormal, "CostNormal");

#[pyclass(name = "CostAR", module = "rustures._rustures")]
struct PyCostAR {
    order: usize,
    inner: Option<RustCostModel>,
}
configured_cost_methods!(PyCostAR, "CostAR");

#[pyclass(name = "CostMahalanobis", module = "rustures._rustures")]
struct PyCostMahalanobis {
    metric: (Vec<f64>, usize, usize),
    inner: Option<RustCostModel>,
}
configured_cost_methods!(PyCostMahalanobis, "CostMahalanobis");

#[pymethods]
impl PyCostNormal {
    #[new]
    #[pyo3(signature = (ridge = 1.0e-6))]
    fn new(ridge: f64) -> PyResult<Self> {
        catch_panic("constructing CostNormal", || {
            if !ridge.is_finite() || ridge <= 0.0 {
                return Err(Error::InvalidRidge { value: ridge }.into());
            }
            Ok(Self { ridge, inner: None })
        })
    }
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting CostNormal", || {
            slf.inner = Some(cost_model_from_numpy(
                signal,
                &RustCostSpec::Normal { ridge: slf.ridge },
            )?);
            Ok(slf)
        })
    }
    fn error(&self, start: usize, end: usize) -> PyResult<f64> {
        catch_panic("computing a CostNormal segment", || {
            Ok(self.fitted()?.cost(start..end)?)
        })
    }
    fn sum_of_costs(&self, breakpoints: Vec<usize>) -> PyResult<f64> {
        catch_panic("summing CostNormal segments", || {
            Ok(sum_model_costs(self.fitted()?, &breakpoints)?)
        })
    }
    #[getter]
    fn ridge(&self) -> f64 {
        self.ridge
    }
    #[getter]
    fn n_samples(&self) -> PyResult<usize> {
        catch_panic("reading CostNormal.n_samples", || {
            Ok(self.fitted()?.n_samples())
        })
    }
    #[getter]
    fn n_features(&self) -> PyResult<usize> {
        catch_panic("reading CostNormal.n_features", || {
            Ok(self.fitted()?.n_features())
        })
    }
    #[getter]
    fn min_size(&self) -> usize {
        2
    }
    #[getter]
    fn is_fitted(&self) -> bool {
        self.inner.is_some()
    }
    fn __repr__(&self) -> String {
        format!("CostNormal(ridge={:?})", self.ridge)
    }
}

#[pymethods]
impl PyCostAR {
    #[new]
    #[pyo3(signature = (order = 4))]
    fn new(order: usize) -> PyResult<Self> {
        catch_panic("constructing CostAR", || {
            if order == 0 {
                return Err(Error::InvalidOrder { value: order }.into());
            }
            Ok(Self { order, inner: None })
        })
    }
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting CostAR", || {
            slf.inner = Some(cost_model_from_numpy(
                signal,
                &RustCostSpec::AR { order: slf.order },
            )?);
            Ok(slf)
        })
    }
    fn error(&self, start: usize, end: usize) -> PyResult<f64> {
        catch_panic("computing a CostAR segment", || {
            Ok(self.fitted()?.cost(start..end)?)
        })
    }
    fn sum_of_costs(&self, breakpoints: Vec<usize>) -> PyResult<f64> {
        catch_panic("summing CostAR segments", || {
            Ok(sum_model_costs(self.fitted()?, &breakpoints)?)
        })
    }
    #[getter]
    fn order(&self) -> usize {
        self.order
    }
    #[getter]
    fn n_samples(&self) -> PyResult<usize> {
        catch_panic("reading CostAR.n_samples", || {
            Ok(self.fitted()?.n_samples())
        })
    }
    #[getter]
    fn n_features(&self) -> PyResult<usize> {
        catch_panic("reading CostAR.n_features", || {
            Ok(self.fitted()?.n_features())
        })
    }
    #[getter]
    fn min_size(&self) -> usize {
        5usize.max(self.order.saturating_add(1))
    }
    #[getter]
    fn is_fitted(&self) -> bool {
        self.inner.is_some()
    }
    fn __repr__(&self) -> String {
        format!("CostAR(order={})", self.order)
    }
}

#[pymethods]
impl PyCostMahalanobis {
    #[new]
    fn new(metric: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        catch_panic("constructing CostMahalanobis", || {
            let shape = metric.shape();
            let metric = (
                metric.as_array().iter().copied().collect(),
                shape[0],
                shape[1],
            );
            Ok(Self {
                metric,
                inner: None,
            })
        })
    }
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting CostMahalanobis", || {
            slf.inner = Some(cost_model_from_numpy(
                signal,
                &RustCostSpec::Mahalanobis {
                    metric: Some(slf.metric.clone()),
                },
            )?);
            Ok(slf)
        })
    }
    fn error(&self, start: usize, end: usize) -> PyResult<f64> {
        catch_panic("computing a CostMahalanobis segment", || {
            Ok(self.fitted()?.cost(start..end)?)
        })
    }
    fn sum_of_costs(&self, breakpoints: Vec<usize>) -> PyResult<f64> {
        catch_panic("summing CostMahalanobis segments", || {
            Ok(sum_model_costs(self.fitted()?, &breakpoints)?)
        })
    }
    #[getter]
    fn metric_dimension(&self) -> usize {
        self.metric.1
    }
    #[getter]
    fn n_samples(&self) -> PyResult<usize> {
        catch_panic("reading CostMahalanobis.n_samples", || {
            Ok(self.fitted()?.n_samples())
        })
    }
    #[getter]
    fn n_features(&self) -> PyResult<usize> {
        catch_panic("reading CostMahalanobis.n_features", || {
            Ok(self.fitted()?.n_features())
        })
    }
    #[getter]
    fn min_size(&self) -> usize {
        2
    }
    #[getter]
    fn is_fitted(&self) -> bool {
        self.inner.is_some()
    }
    fn __repr__(&self) -> String {
        format!("CostMahalanobis(metric_dimension={})", self.metric.1)
    }
}

/// Exact fixed-change dynamic programming detector.
#[pyclass(name = "Dynp", module = "rustures._rustures")]
struct PyDynp {
    model: String,
    spec: RustCostSpec,
    detector: RustDynp,
    cost: Option<Arc<RustCostModel>>,
    custom_source: Option<Py<PyAny>>,
    custom_cost: Option<Arc<PythonSegmentCost>>,
}

/// Exact penalized change-point detection using PELT pruning.
#[pyclass(name = "Pelt", module = "rustures._rustures")]
struct PyPelt {
    model: String,
    spec: RustCostSpec,
    detector: RustPelt,
    cost: Option<Arc<RustCostModel>>,
    custom_source: Option<Py<PyAny>>,
    custom_cost: Option<Arc<PythonSegmentCost>>,
}

#[pyclass(name = "Binseg", module = "rustures._rustures")]
struct PyBinseg {
    model: String,
    spec: RustCostSpec,
    detector: RustBinseg,
    cost: Option<Arc<RustCostModel>>,
}

#[pyclass(name = "BottomUp", module = "rustures._rustures")]
struct PyBottomUp {
    model: String,
    spec: RustCostSpec,
    detector: RustBottomUp,
    cost: Option<Arc<RustCostModel>>,
}

#[pyclass(name = "Window", module = "rustures._rustures")]
struct PyWindow {
    model: String,
    spec: RustCostSpec,
    detector: RustWindow,
    cost: Option<Arc<RustCostModel>>,
}

/// Exact scalar weighted L1-Potts detector with O(KN) dynamic programming.
#[pyclass(name = "L1Potts", module = "rustures._rustures")]
struct PyL1Potts {
    solver: Option<Arc<RustL1Potts>>,
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

fn model_spec(model: &str) -> Result<(String, RustCostSpec), Error> {
    let model = model.to_ascii_lowercase();
    let spec = match model.as_str() {
        "l2" => RustCostSpec::L2,
        "l1" => RustCostSpec::L1,
        "rank" => RustCostSpec::Rank,
        "normal" => RustCostSpec::Normal { ridge: 1.0e-6 },
        "linear" => RustCostSpec::Linear,
        "ar" => RustCostSpec::AR { order: 4 },
        "clinear" => RustCostSpec::CLinear,
        "mahalanobis" | "ml" => RustCostSpec::Mahalanobis { metric: None },
        _ => return Err(Error::UnsupportedModel { model }),
    };
    Ok((model, spec))
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
    fn fitted_n_samples(&self) -> Result<usize, Error> {
        if let Some(custom) = &self.custom_cost {
            return Ok(custom.n_samples());
        }
        self.cost
            .as_ref()
            .map(|cost| cost.n_samples())
            .ok_or(Error::NotFitted { object: "Dynp" })
    }

    fn predict_inner(&self, py: Python<'_>, n_bkps: usize) -> PyResult<Vec<usize>> {
        if let Some(custom) = &self.custom_cost {
            custom.take_callback_error();
            let result = self.detector.predict_changes(custom.as_ref(), n_bkps);
            if let Some(error) = custom.take_callback_error() {
                return Err(error);
            }
            return Ok(result?.breakpoints);
        }

        let cost = Arc::clone(
            self.cost
                .as_ref()
                .ok_or(Error::NotFitted { object: "Dynp" })?,
        );
        let detector = self.detector;
        Ok(py
            .detach(move || detector.predict_changes(cost.as_ref(), n_bkps))?
            .breakpoints)
    }
}

impl PyPelt {
    fn predict_inner(&self, py: Python<'_>, pen: f64) -> PyResult<Vec<usize>> {
        if let Some(custom) = &self.custom_cost {
            custom.take_callback_error();
            let result = self.detector.predict_penalty(custom.as_ref(), pen);
            if let Some(error) = custom.take_callback_error() {
                return Err(error);
            }
            return Ok(result?.breakpoints);
        }

        let cost = Arc::clone(
            self.cost
                .as_ref()
                .ok_or(Error::NotFitted { object: "Pelt" })?,
        );
        let detector = self.detector;
        Ok(py
            .detach(move || detector.predict_penalty(cost.as_ref(), pen))?
            .breakpoints)
    }
}

macro_rules! fitted_cost {
    ($name:ident, $label:literal) => {
        impl $name {
            fn fitted(&self) -> Result<&Arc<RustCostModel>, Error> {
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
    #[pyo3(signature = (model = "l2", custom_cost = None, min_size = 2, jump = 5, max_memory_bytes = 536_870_912))]
    fn new(
        py: Python<'_>,
        model: &str,
        custom_cost: Option<Py<PyAny>>,
        min_size: usize,
        jump: usize,
        max_memory_bytes: usize,
    ) -> PyResult<Self> {
        catch_panic("constructing Dynp", || {
            validate_min_size(min_size)?;
            let (model, spec, min_size) = if let Some(custom) = &custom_cost {
                let cost_min_size = PythonSegmentCost::protocol_min_size(py, custom)?;
                (
                    "custom".to_owned(),
                    RustCostSpec::L2,
                    min_size.max(cost_min_size),
                )
            } else {
                let (model, spec) = model_spec(model)?;
                let min_size = min_size.max(spec.minimum_size_hint());
                (model, spec, min_size)
            };
            Ok(Self {
                model,
                spec,
                detector: RustDynp::with_memory_limit(min_size, jump, max_memory_bytes)?,
                cost: None,
                custom_source: custom_cost,
                custom_cost: None,
            })
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting Dynp", || {
            slf.cost = None;
            slf.custom_cost = None;
            if let Some(source) = &slf.custom_source {
                slf.custom_cost = Some(custom_cost_from_numpy(
                    signal.py(),
                    &signal,
                    source,
                    slf.detector.grid().min_size,
                    "Dynp",
                )?);
            } else {
                slf.cost = Some(Arc::new(cost_model_from_numpy(signal, &slf.spec)?));
            }
            Ok(slf)
        })
    }

    fn predict(&self, py: Python<'_>, n_bkps: usize) -> PyResult<Vec<usize>> {
        catch_panic("predicting with Dynp", || self.predict_inner(py, n_bkps))
    }

    fn estimated_memory_bytes(&self, n_bkps: usize) -> PyResult<usize> {
        catch_panic("estimating Dynp prediction memory", || {
            Ok(self
                .detector
                .estimated_memory_bytes(self.fitted_n_samples()?, n_bkps)?)
        })
    }

    fn fit_predict(
        &mut self,
        py: Python<'_>,
        signal: PyReadonlyArrayDyn<'_, f64>,
        n_bkps: usize,
    ) -> PyResult<Vec<usize>> {
        catch_panic("fitting and predicting with Dynp", || {
            self.cost = None;
            self.custom_cost = None;
            if let Some(source) = &self.custom_source {
                self.custom_cost = Some(custom_cost_from_numpy(
                    py,
                    &signal,
                    source,
                    self.detector.grid().min_size,
                    "Dynp",
                )?);
            } else {
                self.cost = Some(Arc::new(cost_model_from_numpy(signal, &self.spec)?));
            }
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
    fn max_memory_bytes(&self) -> PyResult<usize> {
        catch_panic("reading Dynp.max_memory_bytes", || {
            Ok(self.detector.max_memory_bytes())
        })
    }

    #[getter]
    fn is_fitted(&self) -> PyResult<bool> {
        catch_panic("reading Dynp.is_fitted", || {
            Ok(self.cost.is_some() || self.custom_cost.is_some())
        })
    }

    #[getter]
    fn uses_custom_cost(&self) -> PyResult<bool> {
        catch_panic("reading Dynp.uses_custom_cost", || {
            Ok(self.custom_source.is_some())
        })
    }

    #[getter]
    fn uses_batch_callback(&self) -> PyResult<bool> {
        catch_panic("reading Dynp.uses_batch_callback", || {
            Ok(self
                .custom_cost
                .as_ref()
                .is_some_and(|cost| cost.uses_batch_callback()))
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        catch_panic("formatting Dynp", || {
            Ok(format!(
                "Dynp(model={:?}, min_size={}, jump={}, max_memory_bytes={}, custom_cost={})",
                self.model,
                self.detector.grid().min_size,
                self.detector.grid().jump,
                self.detector.max_memory_bytes(),
                self.custom_source.is_some()
            ))
        })
    }
}

#[pymethods]
impl PyPelt {
    #[new]
    #[pyo3(signature = (model = "l2", custom_cost = None, min_size = 2, jump = 5))]
    fn new(
        py: Python<'_>,
        model: &str,
        custom_cost: Option<Py<PyAny>>,
        min_size: usize,
        jump: usize,
    ) -> PyResult<Self> {
        catch_panic("constructing Pelt", || {
            validate_min_size(min_size)?;
            let (model, spec, min_size) = if let Some(custom) = &custom_cost {
                let cost_min_size = PythonSegmentCost::protocol_min_size(py, custom)?;
                (
                    "custom".to_owned(),
                    RustCostSpec::L2,
                    min_size.max(cost_min_size),
                )
            } else {
                let (model, spec) = model_spec(model)?;
                let min_size = min_size.max(spec.minimum_size_hint());
                (model, spec, min_size)
            };
            Ok(Self {
                model,
                spec,
                detector: RustPelt::new(min_size, jump)?,
                cost: None,
                custom_source: custom_cost,
                custom_cost: None,
            })
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting Pelt", || {
            slf.cost = None;
            slf.custom_cost = None;
            if let Some(source) = &slf.custom_source {
                slf.custom_cost = Some(custom_cost_from_numpy(
                    signal.py(),
                    &signal,
                    source,
                    slf.detector.grid().min_size,
                    "Pelt",
                )?);
            } else {
                slf.cost = Some(Arc::new(cost_model_from_numpy(signal, &slf.spec)?));
            }
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
            self.cost = None;
            self.custom_cost = None;
            if let Some(source) = &self.custom_source {
                self.custom_cost = Some(custom_cost_from_numpy(
                    py,
                    &signal,
                    source,
                    self.detector.grid().min_size,
                    "Pelt",
                )?);
            } else {
                self.cost = Some(Arc::new(cost_model_from_numpy(signal, &self.spec)?));
            }
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
        catch_panic("reading Pelt.is_fitted", || {
            Ok(self.cost.is_some() || self.custom_cost.is_some())
        })
    }

    #[getter]
    fn uses_custom_cost(&self) -> PyResult<bool> {
        catch_panic("reading Pelt.uses_custom_cost", || {
            Ok(self.custom_source.is_some())
        })
    }

    #[getter]
    fn uses_batch_callback(&self) -> PyResult<bool> {
        catch_panic("reading Pelt.uses_batch_callback", || {
            Ok(self
                .custom_cost
                .as_ref()
                .is_some_and(|cost| cost.uses_batch_callback()))
        })
    }

    #[getter]
    fn uses_pelt_pruning(&self) -> PyResult<bool> {
        catch_panic("reading Pelt.uses_pelt_pruning", || {
            if let Some(cost) = &self.custom_cost {
                Ok(cost.uses_pelt_pruning())
            } else {
                Ok(self
                    .cost
                    .as_ref()
                    .is_some_and(|cost| cost.pelt_pruning_constant().is_some()))
            }
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        catch_panic("formatting Pelt", || {
            Ok(format!(
                "Pelt(model={:?}, min_size={}, jump={}, custom_cost={})",
                self.model,
                self.detector.grid().min_size,
                self.detector.grid().jump,
                self.custom_source.is_some()
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
            let (model, spec) = model_spec(model)?;
            validate_min_size(min_size)?;
            let min_size = min_size.max(spec.minimum_size_hint());
            Ok(Self {
                model,
                spec,
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
            slf.cost = Some(Arc::new(cost_model_from_numpy(signal, &slf.spec)?));
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
            self.cost = Some(Arc::new(cost_model_from_numpy(signal, &self.spec)?));
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
            let (model, spec) = model_spec(model)?;
            validate_min_size(min_size)?;
            let min_size = min_size.max(spec.minimum_size_hint());
            Ok(Self {
                model,
                spec,
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
            slf.cost = Some(Arc::new(cost_model_from_numpy(signal, &slf.spec)?));
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
            self.cost = Some(Arc::new(cost_model_from_numpy(signal, &self.spec)?));
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
            let (model, spec) = model_spec(model)?;
            validate_min_size(min_size)?;
            let min_size = min_size.max(spec.minimum_size_hint());
            Ok(Self {
                model,
                spec,
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
            slf.cost = Some(Arc::new(cost_model_from_numpy(signal, &slf.spec)?));
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
            self.cost = Some(Arc::new(cost_model_from_numpy(signal, &self.spec)?));
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
impl PyL1Potts {
    #[new]
    fn new() -> Self {
        Self { solver: None }
    }

    #[pyo3(signature = (signal, weights = None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        signal: PyReadonlyArrayDyn<'py, f64>,
        weights: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        catch_panic("fitting L1Potts", || {
            slf.solver = Some(Arc::new(l1_potts_from_numpy(signal, weights)?));
            Ok(slf)
        })
    }

    fn predict(&self, py: Python<'_>, pen: f64) -> PyResult<Vec<usize>> {
        catch_panic("predicting with L1Potts", || {
            let solver = Arc::clone(
                self.solver
                    .as_ref()
                    .ok_or(Error::NotFitted { object: "L1Potts" })?,
            );
            Ok(py.detach(move || solver.predict_penalty(pen))?.breakpoints)
        })
    }

    #[pyo3(signature = (signal, pen, weights = None))]
    fn fit_predict(
        &mut self,
        py: Python<'_>,
        signal: PyReadonlyArrayDyn<'_, f64>,
        pen: f64,
        weights: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Vec<usize>> {
        catch_panic("fitting and predicting with L1Potts", || {
            self.solver = Some(Arc::new(l1_potts_from_numpy(signal, weights)?));
            self.predict(py, pen)
        })
    }

    #[getter]
    fn n_samples(&self) -> PyResult<usize> {
        catch_panic("reading L1Potts.n_samples", || {
            Ok(self
                .solver
                .as_ref()
                .ok_or(Error::NotFitted { object: "L1Potts" })?
                .n_samples())
        })
    }

    #[getter]
    fn distinct_levels(&self) -> PyResult<usize> {
        catch_panic("reading L1Potts.distinct_levels", || {
            Ok(self
                .solver
                .as_ref()
                .ok_or(Error::NotFitted { object: "L1Potts" })?
                .distinct_levels())
        })
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.solver.is_some()
    }

    fn __repr__(&self) -> String {
        match &self.solver {
            Some(solver) => format!(
                "L1Potts(n_samples={}, distinct_levels={})",
                solver.n_samples(),
                solver.distinct_levels()
            ),
            None => "L1Potts()".to_owned(),
        }
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

#[pymodule]
fn _rustures(module: &Bound<'_, PyModule>) -> PyResult<()> {
    catch_panic("initializing the rustures extension", || {
        module.add("RusturesError", module.py().get_type::<RusturesError>())?;
        module.add("__version__", env!("CARGO_PKG_VERSION"))?;
        module.add_class::<PyCostL2>()?;
        module.add_class::<PyCostL1>()?;
        module.add_class::<PyCostRank>()?;
        module.add_class::<PyCostNormal>()?;
        module.add_class::<PyCostLinear>()?;
        module.add_class::<PyCostAR>()?;
        module.add_class::<PyCostCLinear>()?;
        module.add_class::<PyCostMahalanobis>()?;
        module.add("CostMl", module.py().get_type::<PyCostMahalanobis>())?;
        module.add_class::<PyDynp>()?;
        module.add_class::<PyPelt>()?;
        module.add_class::<PyBinseg>()?;
        module.add_class::<PyBottomUp>()?;
        module.add_class::<PyWindow>()?;
        module.add_class::<PyL1Potts>()?;
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
        test_support::register(module)?;
        Ok(())
    })
}
