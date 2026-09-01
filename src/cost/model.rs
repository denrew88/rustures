use std::ops::Range;

use crate::{Error, SegmentCost, SignalView};

use super::{
    CostAR, CostCLinear, CostL1, CostL2, CostLinear, CostMahalanobis, CostNormal, CostRank,
};

#[derive(Clone, Debug)]
pub enum CostSpec {
    L2,
    L1,
    Rank,
    Normal {
        ridge: f64,
    },
    Linear,
    AR {
        order: usize,
    },
    CLinear,
    Mahalanobis {
        metric: Option<(Vec<f64>, usize, usize)>,
    },
}

impl CostSpec {
    pub fn minimum_size_hint(&self) -> usize {
        match self {
            Self::L2 => 1,
            Self::L1
            | Self::Rank
            | Self::Normal { .. }
            | Self::Linear
            | Self::Mahalanobis { .. } => 2,
            Self::AR { order } => 5usize.max(order.saturating_add(1)),
            Self::CLinear => 3,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::L1 => "l1",
            Self::Rank => "rank",
            Self::Normal { .. } => "normal",
            Self::Linear => "linear",
            Self::AR { .. } => "ar",
            Self::CLinear => "clinear",
            Self::Mahalanobis { .. } => "mahalanobis",
        }
    }

    pub fn fit(&self, signal: SignalView<'_>) -> Result<CostModel, Error> {
        Ok(match self {
            Self::L2 => CostModel::L2(CostL2::fit(signal)?),
            Self::L1 => CostModel::L1(CostL1::fit(signal)),
            Self::Rank => CostModel::Rank(CostRank::fit(signal)?),
            Self::Normal { ridge } => CostModel::Normal(CostNormal::fit(signal, *ridge)?),
            Self::Linear => CostModel::Linear(CostLinear::fit(signal)?),
            Self::AR { order } => CostModel::AR(CostAR::fit(signal, *order)?),
            Self::CLinear => CostModel::CLinear(CostCLinear::fit(signal)),
            Self::Mahalanobis { metric } => match metric {
                Some((values, rows, columns)) => CostModel::Mahalanobis(CostMahalanobis::fit(
                    signal,
                    values.clone(),
                    *rows,
                    *columns,
                )?),
                None => CostModel::Mahalanobis(CostMahalanobis::identity(signal)?),
            },
        })
    }
}

#[derive(Clone, Debug)]
pub enum CostModel {
    L2(CostL2),
    L1(CostL1),
    Rank(CostRank),
    Normal(CostNormal),
    Linear(CostLinear),
    AR(CostAR),
    CLinear(CostCLinear),
    Mahalanobis(CostMahalanobis),
}

macro_rules! delegate {
    ($self:ident, $method:ident $(, $argument:expr)?) => {
        match $self {
            Self::L2(cost) => cost.$method($($argument)?),
            Self::L1(cost) => cost.$method($($argument)?),
            Self::Rank(cost) => cost.$method($($argument)?),
            Self::Normal(cost) => cost.$method($($argument)?),
            Self::Linear(cost) => cost.$method($($argument)?),
            Self::AR(cost) => cost.$method($($argument)?),
            Self::CLinear(cost) => cost.$method($($argument)?),
            Self::Mahalanobis(cost) => cost.$method($($argument)?),
        }
    };
}

impl SegmentCost for CostModel {
    fn n_samples(&self) -> usize {
        delegate!(self, n_samples)
    }
    fn n_features(&self) -> usize {
        delegate!(self, n_features)
    }
    fn min_size(&self) -> usize {
        delegate!(self, min_size)
    }
    fn cost(&self, segment: Range<usize>) -> Result<f64, Error> {
        delegate!(self, cost, segment)
    }
    fn costs_ending_at(
        &self,
        starts: &[usize],
        end: usize,
        output: &mut Vec<f64>,
    ) -> Result<(), Error> {
        match self {
            Self::L2(cost) => cost.costs_ending_at(starts, end, output),
            Self::L1(cost) => cost.costs_ending_at(starts, end, output),
            Self::Rank(cost) => cost.costs_ending_at(starts, end, output),
            Self::Normal(cost) => cost.costs_ending_at(starts, end, output),
            Self::Linear(cost) => cost.costs_ending_at(starts, end, output),
            Self::AR(cost) => cost.costs_ending_at(starts, end, output),
            Self::CLinear(cost) => cost.costs_ending_at(starts, end, output),
            Self::Mahalanobis(cost) => cost.costs_ending_at(starts, end, output),
        }
    }
    fn pelt_pruning_constant(&self) -> Option<f64> {
        delegate!(self, pelt_pruning_constant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{oracle, Dynp, Pelt};

    #[test]
    fn every_stage7_cost_matches_exhaustive_dynp_and_pelt() {
        let scalar = [0.0, 0.2, -0.1, 0.1, 0.0, 5.0, 5.2, 4.9, 5.1, 5.0];
        let scalar_view = SignalView::new(&scalar, 10, 1).unwrap();
        let mut cases = vec![
            CostSpec::L1.fit(scalar_view).unwrap(),
            CostSpec::Rank.fit(scalar_view).unwrap(),
            CostSpec::Normal { ridge: 1.0e-6 }.fit(scalar_view).unwrap(),
            CostSpec::AR { order: 1 }.fit(scalar_view).unwrap(),
            CostSpec::CLinear.fit(scalar_view).unwrap(),
            CostSpec::Mahalanobis { metric: None }
                .fit(scalar_view)
                .unwrap(),
        ];
        let mut linear_values = Vec::new();
        for index in 0..10 {
            let x = index as f64;
            let response = if index < 5 { 1.0 + x } else { 15.0 - x };
            linear_values.extend_from_slice(&[response, 1.0, x]);
        }
        cases.push(
            CostSpec::Linear
                .fit(SignalView::new(&linear_values, 10, 3).unwrap())
                .unwrap(),
        );

        for cost in cases {
            let min_size = cost.min_size();
            let expected_fixed =
                oracle::best_fixed_changes(10, min_size, 1, |range| cost.cost(range).unwrap())
                    .unwrap()
                    .unwrap();
            let actual_fixed = Dynp::new(min_size, 1)
                .unwrap()
                .predict_changes(&cost, 1)
                .unwrap();
            assert_eq!(actual_fixed.breakpoints, expected_fixed.0);

            let penalty = 1.5;
            let expected_penalized =
                oracle::best_penalized(10, min_size, penalty, |range| cost.cost(range).unwrap())
                    .unwrap();
            let actual_penalized = Pelt::new(min_size, 1)
                .unwrap()
                .predict_penalty(&cost, penalty)
                .unwrap();
            assert_eq!(actual_penalized.breakpoints, expected_penalized.0);
            assert!((actual_penalized.objective - expected_penalized.1).abs() < 1.0e-9);
        }
    }
}
