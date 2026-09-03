use std::ops::Range;

use crate::{candidate_is_better, partition_cost, validate_min_size, validate_penalty, Error};

const MAX_EXHAUSTIVE_SAMPLES: usize = 12;

pub(crate) fn enumerate_partitions(
    n_samples: usize,
    min_size: usize,
) -> Result<Vec<Vec<usize>>, Error> {
    validate_min_size(min_size)?;
    if n_samples > MAX_EXHAUSTIVE_SAMPLES {
        return Err(Error::ExhaustiveLimitExceeded {
            actual: n_samples,
            maximum: MAX_EXHAUSTIVE_SAMPLES,
        });
    }
    if n_samples == 0 {
        return Err(Error::EmptySignal);
    }

    let mut partitions = Vec::new();
    let mut current = Vec::new();
    enumerate_from(0, n_samples, min_size, &mut current, &mut partitions);
    Ok(partitions)
}

fn enumerate_from(
    start: usize,
    n_samples: usize,
    min_size: usize,
    current: &mut Vec<usize>,
    partitions: &mut Vec<Vec<usize>>,
) {
    if n_samples - start < min_size {
        return;
    }

    current.push(n_samples);
    partitions.push(current.clone());
    current.pop();

    let first_end = start + min_size;
    let last_end = n_samples - min_size;
    for end in first_end..=last_end {
        current.push(end);
        enumerate_from(end, n_samples, min_size, current, partitions);
        current.pop();
    }
}

pub(crate) fn best_fixed_changes<F>(
    n_samples: usize,
    min_size: usize,
    changes: usize,
    mut segment_cost: F,
) -> Result<Option<(Vec<usize>, f64)>, Error>
where
    F: FnMut(Range<usize>) -> f64,
{
    let mut best: Option<(Vec<usize>, f64)> = None;
    for breakpoints in enumerate_partitions(n_samples, min_size)? {
        if breakpoints.len() != changes + 1 {
            continue;
        }
        let cost = partition_cost(&breakpoints, &mut segment_cost)?;
        if best.as_ref().is_none_or(|(best_breakpoints, best_cost)| {
            candidate_is_better(cost, &breakpoints, *best_cost, best_breakpoints)
        }) {
            best = Some((breakpoints, cost));
        }
    }
    Ok(best)
}

pub(crate) fn best_penalized<F>(
    n_samples: usize,
    min_size: usize,
    penalty: f64,
    mut segment_cost: F,
) -> Result<(Vec<usize>, f64), Error>
where
    F: FnMut(Range<usize>) -> f64,
{
    validate_penalty(penalty)?;
    let mut best: Option<(Vec<usize>, f64)> = None;
    for breakpoints in enumerate_partitions(n_samples, min_size)? {
        let raw_cost = partition_cost(&breakpoints, &mut segment_cost)?;
        let objective = raw_cost + penalty * (breakpoints.len() - 1) as f64;
        if best.as_ref().is_none_or(|(best_breakpoints, best_cost)| {
            candidate_is_better(objective, &breakpoints, *best_cost, best_breakpoints)
        }) {
            best = Some((breakpoints, objective));
        }
    }
    Ok(best.expect("a non-empty feasible signal always has one partition"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squared_mean_cost(signal: &[f64], range: Range<usize>) -> f64 {
        let segment = &signal[range];
        let mean = segment.iter().sum::<f64>() / segment.len() as f64;
        segment.iter().map(|value| (value - mean).powi(2)).sum()
    }

    #[test]
    fn enumerates_every_partition_for_min_size_one() {
        let partitions = enumerate_partitions(4, 1).unwrap();
        assert_eq!(partitions.len(), 8);
        assert!(partitions.contains(&vec![4]));
        assert!(partitions.contains(&vec![1, 2, 3, 4]));
    }

    #[test]
    fn enforces_minimum_segment_size() {
        assert_eq!(
            enumerate_partitions(5, 2).unwrap(),
            vec![vec![5], vec![2, 5], vec![3, 5]]
        );
    }

    #[test]
    fn finds_best_fixed_partition() {
        let signal = [0.0, 0.0, 10.0, 10.0];
        let best = best_fixed_changes(4, 1, 1, |range| squared_mean_cost(&signal, range))
            .unwrap()
            .unwrap();
        assert_eq!(best, (vec![2, 4], 0.0));
    }

    #[test]
    fn finds_best_penalized_partition() {
        let signal = [0.0, 0.0, 10.0, 10.0];
        let best = best_penalized(4, 1, 1.0, |range| squared_mean_cost(&signal, range)).unwrap();
        assert_eq!(best, (vec![2, 4], 1.0));
    }

    #[test]
    fn rejects_large_exhaustive_problem() {
        assert_eq!(
            enumerate_partitions(13, 1),
            Err(Error::ExhaustiveLimitExceeded {
                actual: 13,
                maximum: 12,
            })
        );
    }
}
