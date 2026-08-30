use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignalShape {
    pub n_samples: usize,
    pub n_features: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct SignalView<'a> {
    values: &'a [f64],
    shape: SignalShape,
}

impl<'a> SignalView<'a> {
    pub fn new(values: &'a [f64], n_samples: usize, n_features: usize) -> Result<Self, Error> {
        let shape = validate_signal_shape(2, &[n_samples, n_features])?;
        let expected = n_samples
            .checked_mul(n_features)
            .ok_or(Error::NumericalFailure {
                context: "computing signal storage size",
            })?;
        if values.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                actual: values.len(),
            });
        }
        validate_finite(values.iter().copied(), shape)?;
        Ok(Self { values, shape })
    }

    pub fn shape(self) -> SignalShape {
        self.shape
    }

    pub fn values(self) -> &'a [f64] {
        self.values
    }

    pub fn row(self, index: usize) -> Option<&'a [f64]> {
        if index >= self.shape.n_samples {
            return None;
        }
        let start = index * self.shape.n_features;
        Some(&self.values[start..start + self.shape.n_features])
    }
}

pub fn validate_signal_shape(ndim: usize, shape: &[usize]) -> Result<SignalShape, Error> {
    let result = match (ndim, shape) {
        (1, [n_samples]) => SignalShape {
            n_samples: *n_samples,
            n_features: 1,
        },
        (2, [n_samples, n_features]) => SignalShape {
            n_samples: *n_samples,
            n_features: *n_features,
        },
        _ => return Err(Error::InvalidSignalDimension { ndim }),
    };

    if result.n_samples == 0 || result.n_features == 0 {
        return Err(Error::EmptySignal);
    }
    Ok(result)
}

pub fn validate_finite(
    values: impl IntoIterator<Item = f64>,
    shape: SignalShape,
) -> Result<(), Error> {
    for (flat_index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::NonFiniteInput {
                row: flat_index / shape.n_features,
                column: flat_index % shape.n_features,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_signal_has_one_feature() {
        assert_eq!(
            validate_signal_shape(1, &[3]),
            Ok(SignalShape {
                n_samples: 3,
                n_features: 1,
            })
        );
    }

    #[test]
    fn rejects_empty_and_high_rank_signals() {
        assert_eq!(validate_signal_shape(1, &[0]), Err(Error::EmptySignal));
        assert_eq!(
            validate_signal_shape(3, &[1, 1, 1]),
            Err(Error::InvalidSignalDimension { ndim: 3 })
        );
    }

    #[test]
    fn reports_non_finite_coordinates() {
        let shape = SignalShape {
            n_samples: 2,
            n_features: 2,
        };
        assert_eq!(
            validate_finite([1.0, 2.0, f64::NAN, 4.0], shape),
            Err(Error::NonFiniteInput { row: 1, column: 0 })
        );
    }

    #[test]
    fn signal_view_borrows_row_major_storage() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let view = SignalView::new(&values, 2, 2).unwrap();
        assert_eq!(view.shape().n_features, 2);
        assert_eq!(view.row(1), Some(&values[2..]));
        assert_eq!(view.row(2), None);
    }

    #[test]
    fn signal_view_rejects_storage_length_mismatch() {
        assert_eq!(
            SignalView::new(&[1.0, 2.0, 3.0], 2, 2).unwrap_err(),
            Error::DimensionMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }
}
