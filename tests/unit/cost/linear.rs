use super::*;

fn assert_batch_matches_scalar(cost: &CostLinear, starts: &[usize], end: usize) {
    let mut batch = Vec::new();
    cost.costs_ending_at(starts, end, &mut batch).unwrap();
    assert_eq!(batch.len(), starts.len());
    for (&start, &actual) in starts.iter().zip(&batch) {
        let expected = cost.cost(start..end).unwrap();
        let tolerance = 1.0e-9 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "segment=[{start}, {end}), actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn exact_linear_relation_has_zero_cost() {
    let values = [3., 1., 1., 5., 1., 2., 7., 1., 3., 9., 1., 4.];
    let cost = CostLinear::fit(SignalView::new(&values, 4, 3).unwrap()).unwrap();
    assert!(cost.cost(0..4).unwrap() < 1e-20);

    let response = [3., 5., 7., 9.];
    let design = [1., 1., 1., 2., 1., 3., 1., 4.];
    let explicit = CostLinear::fit_response_design(
        SignalView::new(&response, 4, 1).unwrap(),
        SignalView::new(&design, 4, 2).unwrap(),
    )
    .unwrap();
    assert!(explicit.cost(0..4).unwrap() < 1e-20);

    let collinear = [1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.];
    let singular = CostLinear::fit(SignalView::new(&collinear, 4, 3).unwrap()).unwrap();
    assert!(singular.cost(0..4).unwrap().is_finite());
}

#[test]
fn endpoint_batch_matches_svd_for_regular_singular_and_large_offset_data() {
    let mut regular = Vec::new();
    for index in 0..30 {
        let x = index as f64 / 7.0;
        let noise = ((index * 11) % 7) as f64 * 0.013;
        regular.extend_from_slice(&[2.0 + 1.5 * x + noise, 1.0, x]);
    }
    let regular = CostLinear::fit(SignalView::new(&regular, 30, 3).unwrap()).unwrap();
    assert_batch_matches_scalar(&regular, &[0, 3, 7, 12, 20], 30);

    let mut singular = Vec::new();
    for index in 0..20 {
        let x = index as f64;
        singular.extend_from_slice(&[3.0 * x + 0.1, x, 2.0 * x]);
    }
    let singular = CostLinear::fit(SignalView::new(&singular, 20, 3).unwrap()).unwrap();
    assert_batch_matches_scalar(&singular, &[0, 2, 5, 10], 20);

    let mut offset = Vec::new();
    for index in 0..20 {
        let x = 1.0e12 + index as f64;
        offset.extend_from_slice(&[5.0 + 0.25 * x, 1.0, x]);
    }
    let offset = CostLinear::fit(SignalView::new(&offset, 20, 3).unwrap()).unwrap();
    assert_batch_matches_scalar(&offset, &[0, 2, 6, 12], 20);
}
