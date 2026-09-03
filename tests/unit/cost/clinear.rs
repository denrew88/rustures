use super::*;
#[test]
fn straight_line_has_zero_cost() {
    let cost = CostCLinear::fit(SignalView::new(&[1., 3., 5., 7.], 4, 1).unwrap());
    assert_eq!(cost.cost(0..4).unwrap(), 0.0);
}
