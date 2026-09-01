use std::ops::Range;

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::{validate_segment, Error, SignalShape, SignalView};

const ROUNDOFF_FACTOR: f64 = 128.0;

#[derive(Clone, Debug)]
pub(super) struct ScatterStats {
    pub shape: SignalShape,
    prefix_sum: Vec<f64>,
    prefix_outer: Vec<f64>,
}

impl ScatterStats {
    pub fn fit(signal: SignalView<'_>) -> Result<Self, Error> {
        let shape = signal.shape();
        let d = shape.n_features;
        let rows = shape.n_samples + 1;
        let mut prefix_sum = vec![0.0; rows * d];
        let mut prefix_outer = vec![0.0; rows * d * d];
        let offset = signal.row(0).ok_or(Error::NumericalFailure {
            context: "reading a scatter centering reference",
        })?;
        let mut centered = vec![0.0; d];

        for row in 0..shape.n_samples {
            let values = signal.row(row).ok_or(Error::NumericalFailure {
                context: "reading a scatter observation",
            })?;
            for feature in 0..d {
                centered[feature] = values[feature] - offset[feature];
                if !centered[feature].is_finite() {
                    return Err(Error::NumericalFailure {
                        context: "centering scatter observations",
                    });
                }
                prefix_sum[(row + 1) * d + feature] =
                    prefix_sum[row * d + feature] + centered[feature];
            }
            for left in 0..d {
                for right in 0..d {
                    let index = left * d + right;
                    let value = centered[left] * centered[right];
                    if !value.is_finite() {
                        return Err(Error::NumericalFailure {
                            context: "forming scatter outer products",
                        });
                    }
                    prefix_outer[(row + 1) * d * d + index] =
                        prefix_outer[row * d * d + index] + value;
                }
            }
        }
        Ok(Self {
            shape,
            prefix_sum,
            prefix_outer,
        })
    }

    pub fn sum(&self, segment: Range<usize>, min_size: usize) -> Result<DVector<f64>, Error> {
        validate_segment(segment.clone(), self.shape.n_samples, min_size)?;
        let d = self.shape.n_features;
        Ok(DVector::from_fn(d, |feature, _| {
            self.prefix_sum[segment.end * d + feature]
                - self.prefix_sum[segment.start * d + feature]
        }))
    }

    pub fn scatter(&self, segment: Range<usize>, min_size: usize) -> Result<DMatrix<f64>, Error> {
        validate_segment(segment.clone(), self.shape.n_samples, min_size)?;
        let d = self.shape.n_features;
        let length = segment.len() as f64;
        let sum = self.sum(segment.clone(), min_size)?;
        let mut result = DMatrix::zeros(d, d);
        for left in 0..d {
            for right in 0..d {
                let index = left * d + right;
                let outer = self.prefix_outer[segment.end * d * d + index]
                    - self.prefix_outer[segment.start * d * d + index];
                result[(left, right)] = outer - sum[left] * sum[right] / length;
            }
        }
        for left in 0..d {
            for right in 0..left {
                let average = 0.5 * (result[(left, right)] + result[(right, left)]);
                result[(left, right)] = average;
                result[(right, left)] = average;
            }
        }
        Ok(result)
    }
}

pub(super) fn clamp_nonnegative(value: f64, scale: f64) -> Result<f64, Error> {
    if !value.is_finite() {
        return Err(Error::NonFiniteObjective { value });
    }
    if value >= 0.0 {
        return Ok(value);
    }
    let tolerance = ROUNDOFF_FACTOR * f64::EPSILON * scale.abs().max(1.0);
    if value >= -tolerance {
        Ok(0.0)
    } else {
        Err(Error::NumericalFailure {
            context: "evaluating a non-negative scatter cost",
        })
    }
}

pub(super) fn symmetric_pseudoinverse(matrix: DMatrix<f64>) -> Result<DMatrix<f64>, Error> {
    let dimension = matrix.nrows();
    let decomposition = SymmetricEigen::new(matrix);
    let scale = decomposition
        .eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    let tolerance = ROUNDOFF_FACTOR * f64::EPSILON * dimension.max(1) as f64 * scale;
    if decomposition
        .eigenvalues
        .iter()
        .any(|&value| value < -tolerance || !value.is_finite())
    {
        return Err(Error::NonPositiveSemidefiniteMetric);
    }
    let inverse = DMatrix::from_diagonal(&decomposition.eigenvalues.map(|value| {
        if value > tolerance {
            value.recip()
        } else {
            0.0
        }
    }));
    Ok(&decomposition.eigenvectors * inverse * decomposition.eigenvectors.transpose())
}

pub(super) fn validate_psd(matrix: &DMatrix<f64>) -> Result<(), Error> {
    if matrix.nrows() != matrix.ncols() || matrix.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonPositiveSemidefiniteMetric);
    }
    let scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    let tolerance = ROUNDOFF_FACTOR * f64::EPSILON * matrix.nrows().max(1) as f64 * scale;
    for row in 0..matrix.nrows() {
        for column in 0..row {
            if (matrix[(row, column)] - matrix[(column, row)]).abs() > tolerance {
                return Err(Error::NonPositiveSemidefiniteMetric);
            }
        }
    }
    let eigen = SymmetricEigen::new(matrix.clone());
    if eigen
        .eigenvalues
        .iter()
        .any(|&value| !value.is_finite() || value < -tolerance)
    {
        Err(Error::NonPositiveSemidefiniteMetric)
    } else {
        Ok(())
    }
}
