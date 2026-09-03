use super::*;

fn assert_batch_matches_scalar(cost: &CostAR, starts: &[usize], end: usize) {
    let mut batch = Vec::new();
    cost.costs_ending_at(starts, end, &mut batch).unwrap();
    for (&start, &actual) in starts.iter().zip(&batch) {
        let expected = cost.cost(start..end).unwrap();
        let tolerance = 1.0e-8 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "segment=[{start}, {end}), actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn segment_local_ar_fits_a_deterministic_recurrence() {
    let values = [1., 2., 4., 8., 16., 32.];
    let cost = CostAR::fit_segment_local(SignalView::new(&values, 6, 1).unwrap(), 1).unwrap();
    assert!(cost.cost(0..6).unwrap() < 1e-20);
}

#[test]
fn endpoint_batch_matches_svd_for_both_boundaries_and_multivariate_data() {
    let mut values = Vec::new();
    for index in 0..30 {
        let time = index as f64;
        values.extend_from_slice(&[
            (time * 0.31).sin() + 0.02 * time,
            (time * 0.17).cos() - 0.01 * time,
        ]);
    }
    let signal = SignalView::new(&values, 30, 2).unwrap();
    let compatibility = CostAR::fit(signal, 2).unwrap();
    assert_batch_matches_scalar(&compatibility, &[0, 4, 9, 15, 22], 30);

    let segment_local = CostAR::fit_segment_local(signal, 2).unwrap();
    assert_batch_matches_scalar(&segment_local, &[0, 4, 9, 15, 22], 30);
}
