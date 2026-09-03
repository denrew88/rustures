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
