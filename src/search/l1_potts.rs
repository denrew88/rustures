use crate::{objective_values_tied, validate_penalty, Error, Segmentation, SignalView};

/// Exact scalar weighted L1-Potts solver.
///
/// The state space is the sorted set of distinct observations. The forward
/// pass uses two score rows and one compact byte per sample/state parent.
#[derive(Clone, Debug)]
pub struct L1Potts {
    values: Vec<f64>,
    weights: Vec<f64>,
    levels: Vec<f64>,
}

impl L1Potts {
    pub fn fit(signal: SignalView<'_>, weights: Option<&[f64]>) -> Result<Self, Error> {
        if signal.shape().n_features != 1 {
            return Err(Error::ScalarSignalRequired {
                model: "L1Potts",
                actual: signal.shape().n_features,
            });
        }
        let values = signal.values().to_vec();
        let weights = match weights {
            Some(weights) => {
                if weights.len() != values.len() {
                    return Err(Error::InvalidWeightsLength {
                        expected: values.len(),
                        actual: weights.len(),
                    });
                }
                for (position, &value) in weights.iter().enumerate() {
                    if !value.is_finite() || value < 0.0 {
                        return Err(Error::InvalidWeight { position, value });
                    }
                }
                weights.to_vec()
            }
            None => vec![1.0; values.len()],
        };
        let mut levels = values.clone();
        levels.sort_by(f64::total_cmp);
        levels.dedup();
        Ok(Self {
            values,
            weights,
            levels,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.values.len()
    }

    pub fn distinct_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn predict_penalty(&self, penalty: f64) -> Result<Segmentation, Error> {
        validate_penalty(penalty)?;
        let n = self.values.len();
        let k = self.levels.len();
        let mut previous = vec![0.0; k];
        let mut current = vec![0.0; k];
        for (state, score) in previous.iter_mut().enumerate() {
            *score = self.observation_cost(0, state)?;
        }

        let mut best_states = vec![0usize; n];
        best_states[0] = argmin(&previous);
        let parent_bytes = n.checked_mul(k).ok_or(Error::NumericalFailure {
            context: "allocating the L1-Potts parent matrix",
        })?;
        let mut jumped = vec![0u8; parent_bytes];

        for sample in 1..n {
            let jump_state = best_states[sample - 1];
            let jump_score = previous[jump_state] + penalty;
            if !jump_score.is_finite() {
                return Err(Error::NonFiniteObjective { value: jump_score });
            }
            for state in 0..k {
                let stay_score = previous[state];
                let take_jump =
                    jump_score < stay_score && !objective_values_tied(jump_score, stay_score);
                let predecessor = if take_jump { jump_score } else { stay_score };
                current[state] = predecessor + self.observation_cost(sample, state)?;
                if !current[state].is_finite() {
                    return Err(Error::NonFiniteObjective {
                        value: current[state],
                    });
                }
                jumped[sample * k + state] = u8::from(take_jump);
            }
            best_states[sample] = argmin(&current);
            std::mem::swap(&mut previous, &mut current);
        }

        let mut states = vec![0usize; n];
        let mut state = best_states[n - 1];
        states[n - 1] = state;
        for sample in (1..n).rev() {
            if jumped[sample * k + state] != 0 {
                state = best_states[sample - 1];
            }
            states[sample - 1] = state;
        }

        let mut breakpoints = Vec::new();
        for sample in 1..n {
            if states[sample] != states[sample - 1] {
                breakpoints.push(sample);
            }
        }
        breakpoints.push(n);
        // Recompute the returned data term independently from the DP states,
        // using a weighted median for every reconstructed segment.
        let mut segment_cost = 0.0;
        let mut start = 0;
        for &end in &breakpoints {
            segment_cost += self.weighted_segment_cost(start, end)?;
            start = end;
        }
        let changes = breakpoints.len() - 1;
        let objective = segment_cost + penalty * changes as f64;
        let dp_objective = previous[best_states[n - 1]];
        if !objective_values_tied(objective, dp_objective) {
            return Err(Error::NumericalFailure {
                context: "recomputing the L1-Potts weighted-median objective",
            });
        }
        Segmentation::new(breakpoints, segment_cost, objective, n, 1)
    }

    fn observation_cost(&self, sample: usize, state: usize) -> Result<f64, Error> {
        let value = self.weights[sample] * (self.levels[state] - self.values[sample]).abs();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NonFiniteObjective { value })
        }
    }

    fn weighted_segment_cost(&self, start: usize, end: usize) -> Result<f64, Error> {
        let mut weights_by_level = vec![0.0; self.levels.len()];
        for sample in start..end {
            let level = self
                .levels
                .binary_search_by(|value| value.total_cmp(&self.values[sample]))
                .map_err(|_| Error::NumericalFailure {
                    context: "locating an observation in the L1-Potts state space",
                })?;
            weights_by_level[level] += self.weights[sample];
        }
        let total_weight = weights_by_level.iter().sum::<f64>();
        let median = if total_weight == 0.0 {
            self.values[start]
        } else {
            let mut cumulative = 0.0;
            weights_by_level
                .iter()
                .enumerate()
                .find_map(|(level, &weight)| {
                    cumulative += weight;
                    (2.0 * cumulative >= total_weight).then_some(self.levels[level])
                })
                .unwrap_or(self.levels[self.levels.len() - 1])
        };
        let value = (start..end)
            .map(|sample| self.weights[sample] * (self.values[sample] - median).abs())
            .sum::<f64>();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NonFiniteObjective { value })
        }
    }
}

fn argmin(scores: &[f64]) -> usize {
    let mut best = 0;
    for candidate in 1..scores.len() {
        if scores[candidate] < scores[best]
            && !objective_values_tied(scores[candidate], scores[best])
        {
            best = candidate;
        }
    }
    best
}

#[cfg(test)]
#[path = "../../tests/unit/search/l1_potts.rs"]
mod tests;
