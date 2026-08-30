use crate::{validate_breakpoints, Error};

fn validate_pair<'a>(truth: &'a [usize], prediction: &'a [usize]) -> Result<usize, Error> {
    let truth_n = *truth.last().ok_or(Error::EmptyBreakpoints)?;
    let prediction_n = *prediction.last().ok_or(Error::EmptyBreakpoints)?;
    if truth_n != prediction_n {
        return Err(Error::BreakpointLengthMismatch {
            expected: truth_n,
            actual: prediction_n,
        });
    }
    validate_breakpoints(truth, truth_n, 1)?;
    validate_breakpoints(prediction, prediction_n, 1)?;
    Ok(truth_n)
}

fn directed_hausdorff(left: &[usize], right: &[usize], fallback: usize) -> usize {
    if left.is_empty() {
        return 0;
    }
    if right.is_empty() {
        return fallback;
    }
    left.iter()
        .map(|&point| {
            right
                .iter()
                .map(|&other| point.abs_diff(other))
                .min()
                .unwrap_or(fallback)
        })
        .max()
        .unwrap_or(0)
}

/// Symmetric Hausdorff distance between internal change points.
///
/// The shared terminal sample is excluded. If just one partition has no
/// internal changes, the signal length is returned.
pub fn hausdorff(truth: &[usize], prediction: &[usize]) -> Result<f64, Error> {
    let n_samples = validate_pair(truth, prediction)?;
    let left = &truth[..truth.len() - 1];
    let right = &prediction[..prediction.len() - 1];
    Ok(
        directed_hausdorff(left, right, n_samples).max(directed_hausdorff(right, left, n_samples))
            as f64,
    )
}

/// One-to-one margin matching of predicted and reference change points.
pub fn precision_recall(
    truth: &[usize],
    prediction: &[usize],
    margin: usize,
) -> Result<(f64, f64), Error> {
    validate_pair(truth, prediction)?;
    if margin == 0 {
        return Err(Error::InvalidMargin { value: margin });
    }
    let truth = &truth[..truth.len() - 1];
    let prediction = &prediction[..prediction.len() - 1];
    let mut used = vec![false; prediction.len()];
    let mut matches = 0usize;
    for &reference in truth {
        let best = prediction
            .iter()
            .enumerate()
            .filter(|(index, point)| !used[*index] && reference.abs_diff(**point) < margin)
            .min_by_key(|(index, point)| (reference.abs_diff(**point), **point, *index));
        if let Some((index, _)) = best {
            used[index] = true;
            matches += 1;
        }
    }
    let precision = if prediction.is_empty() {
        f64::from(truth.is_empty())
    } else {
        matches as f64 / prediction.len() as f64
    };
    let recall = if truth.is_empty() {
        1.0
    } else {
        matches as f64 / truth.len() as f64
    };
    Ok((precision, recall))
}

fn pairs(length: usize) -> u128 {
    (length as u128) * (length.saturating_sub(1) as u128) / 2
}

/// Rand index computed from interval intersections without materializing labels
/// or sample pairs.
pub fn rand_index(truth: &[usize], prediction: &[usize]) -> Result<f64, Error> {
    let n_samples = validate_pair(truth, prediction)?;
    if n_samples < 2 {
        return Ok(1.0);
    }

    let same_truth: u128 = truth
        .iter()
        .scan(0usize, |start, &end| {
            let length = end - *start;
            *start = end;
            Some(pairs(length))
        })
        .sum();
    let same_prediction: u128 = prediction
        .iter()
        .scan(0usize, |start, &end| {
            let length = end - *start;
            *start = end;
            Some(pairs(length))
        })
        .sum();

    let (mut truth_index, mut prediction_index) = (0usize, 0usize);
    let (mut truth_start, mut prediction_start) = (0usize, 0usize);
    let mut same_both = 0u128;
    while truth_index < truth.len() && prediction_index < prediction.len() {
        let end = truth[truth_index].min(prediction[prediction_index]);
        let start = truth_start.max(prediction_start);
        same_both += pairs(end - start);
        if truth[truth_index] == end {
            truth_start = end;
            truth_index += 1;
        }
        if prediction[prediction_index] == end {
            prediction_start = end;
            prediction_index += 1;
        }
    }

    let total = pairs(n_samples);
    let different_both = total - same_truth - same_prediction + same_both;
    Ok((same_both + different_both) as f64 / total as f64)
}

#[cfg(test)]
mod tests {
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
}
