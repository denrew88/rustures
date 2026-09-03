use super::*;
use crate::{CostL2, SignalView};

fn cost(values: &[f64]) -> CostL2 {
    CostL2::fit(SignalView::new(values, values.len(), 1).unwrap()).unwrap()
}

#[test]
fn binseg_supports_all_stopping_rules() {
    let cost = cost(&[0., 0., 0., 10., 10., 10., -5., -5., -5.]);
    let detector = Binseg::new(2, 1).unwrap();
    assert_eq!(
        detector
            .predict(&cost, Stop::Changes(2))
            .unwrap()
            .breakpoints,
        [3, 6, 9]
    );
    assert_eq!(
        detector
            .predict(&cost, Stop::Penalty(1.))
            .unwrap()
            .breakpoints,
        [3, 6, 9]
    );
    assert_eq!(
        detector
            .predict(&cost, Stop::Budget(0.))
            .unwrap()
            .breakpoints,
        [3, 6, 9]
    );
}

#[test]
fn bottom_up_uses_adjacency_heap() {
    let cost = cost(&[0., 0., 0., 10., 10., 10., -5., -5., -5.]);
    let detector = BottomUp::new(2, 1).unwrap();
    assert_eq!(
        detector
            .predict(&cost, Stop::Changes(2))
            .unwrap()
            .breakpoints,
        [4, 6, 9]
    );
}

#[test]
fn window_finds_local_discrepancy_peaks() {
    let cost = cost(&[0., 0., 0., 0., 10., 10., 10., 10.]);
    let detector = Window::new(4, 1, 1).unwrap();
    assert_eq!(
        detector
            .predict(&cost, Stop::Changes(1))
            .unwrap()
            .breakpoints,
        [4, 8]
    );
}
