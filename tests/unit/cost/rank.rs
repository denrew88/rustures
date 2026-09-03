use super::*;
#[test]
fn rank_cost_detects_distribution_shift() {
    let values = [0., 1., 2., 10., 11., 12.];
    let cost = CostRank::fit(SignalView::new(&values, 6, 1).unwrap()).unwrap();
    assert!(cost.cost(0..3).unwrap() < cost.cost(1..5).unwrap());
}

#[test]
fn all_tied_ranks_use_a_zero_pseudoinverse_without_nan() {
    let values = [2.0; 8];
    let cost = CostRank::fit(SignalView::new(&values, 8, 1).unwrap()).unwrap();
    assert_eq!(cost.cost(0..4).unwrap(), 0.0);
}
