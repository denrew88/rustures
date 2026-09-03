use super::*;

#[test]
fn metrics_match_small_hand_computed_examples() {
    assert_eq!(hausdorff(&[3, 7, 10], &[2, 8, 10]).unwrap(), 1.0);
    assert_eq!(
        precision_recall(&[3, 7, 10], &[2, 5, 10], 2).unwrap(),
        (0.5, 0.5)
    );
    assert_eq!(rand_index(&[2, 4], &[2, 4]).unwrap(), 1.0);
    assert!((rand_index(&[2, 4], &[1, 3, 4]).unwrap() - 0.5).abs() < 1e-12);
}

#[test]
fn no_change_edge_cases_are_defined() {
    assert_eq!(hausdorff(&[10], &[10]).unwrap(), 0.0);
    assert_eq!(hausdorff(&[10], &[5, 10]).unwrap(), 10.0);
    assert_eq!(precision_recall(&[10], &[10], 1).unwrap(), (1.0, 1.0));
}
