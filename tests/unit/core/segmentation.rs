use super::*;

#[test]
fn accepts_valid_breakpoints_with_terminal_sample() {
    assert_eq!(validate_breakpoints(&[2, 5], 5, 2), Ok(()));
}

#[test]
fn rejects_missing_terminal_breakpoint() {
    assert_eq!(
        validate_breakpoints(&[2, 4], 5, 2),
        Err(Error::MissingTerminalBreakpoint {
            actual: 4,
            n_samples: 5,
        })
    );
}

#[test]
fn rejects_short_segments() {
    assert!(matches!(
        validate_breakpoints(&[1, 5], 5, 2),
        Err(Error::SegmentTooShort { .. })
    ));
}

#[test]
fn tie_breaks_lexicographically() {
    assert!(candidate_is_better(3.0, &[2, 5], 3.0, &[3, 5]));
    assert!(!candidate_is_better(3.0, &[3, 5], 3.0, &[2, 5]));
    assert!(candidate_is_better(
        50.833_333_333_333_336,
        &[1, 2, 8],
        50.833_333_333_333_33,
        &[6, 7, 8],
    ));
}

#[test]
fn specialized_non_negative_comparison_preserves_the_common_tolerance() {
    for incumbent in [0.0, 1.0e-15, 1.0e-6, 1.0, 1.0e6, 1.0e200] {
        let candidates = [
            0.0,
            incumbent * 0.5,
            incumbent * (1.0 - 1.0e-10),
            incumbent,
            incumbent * (1.0 + 1.0e-10),
        ];
        for candidate in candidates {
            let expected_replacement =
                candidate < incumbent && !objective_values_tied(candidate, incumbent);
            if candidate < incumbent {
                assert_eq!(
                    non_negative_candidate_is_significant(candidate, incumbent),
                    expected_replacement,
                    "specialized candidate={candidate}, incumbent={incumbent}"
                );
            } else if candidate > incumbent {
                assert_eq!(
                    non_negative_increase_is_significant(candidate, incumbent),
                    !objective_values_tied(candidate, incumbent),
                    "increase candidate={candidate}, incumbent={incumbent}"
                );
            }
        }
    }
}

#[test]
fn rejects_invalid_parameters() {
    assert_eq!(
        validate_min_size(0),
        Err(Error::InvalidMinSize { value: 0 })
    );
    assert_eq!(validate_jump(0), Err(Error::InvalidJump { value: 0 }));
    assert!(matches!(
        validate_penalty(f64::NAN),
        Err(Error::InvalidPenalty { .. })
    ));
}

#[test]
fn search_grid_and_capabilities_are_explicit() {
    assert_eq!(
        SearchGrid::new(2, 5),
        Ok(SearchGrid {
            min_size: 2,
            jump: 5,
        })
    );
    assert!(DetectorCapabilities::CHANGES_ONLY.supports(Stop::Changes(2)));
    assert!(!DetectorCapabilities::CHANGES_ONLY.supports(Stop::Penalty(1.0)));
}

#[test]
fn segmentation_checks_its_invariants() {
    let result = Segmentation::new(vec![2, 5], 1.5, 3.5, 5, 2).unwrap();
    assert_eq!(result.breakpoints, [2, 5]);
    assert!(matches!(
        Segmentation::new(vec![2, 4], 1.5, 3.5, 5, 2),
        Err(Error::MissingTerminalBreakpoint { .. })
    ));
}
