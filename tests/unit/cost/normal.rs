use super::*;
#[test]
fn regularizes_constant_covariance() {
    let cost = CostNormal::fit(SignalView::new(&[3.; 6], 3, 2).unwrap(), 1e-6).unwrap();
    assert!(cost.cost(0..3).unwrap().is_finite());
}
