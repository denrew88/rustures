use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use crate::{
    objective_values_tied, validate_budget, validate_penalty, Detector, DetectorCapabilities,
    Error, SearchGrid, SegmentCost, Segmentation, Stop,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binseg {
    grid: SearchGrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BottomUp {
    grid: SearchGrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    grid: SearchGrid,
    width: usize,
}

impl Binseg {
    pub fn new(min_size: usize, jump: usize) -> Result<Self, Error> {
        Ok(Self {
            grid: SearchGrid::new(min_size, jump)?,
        })
    }

    pub fn grid(self) -> SearchGrid {
        self.grid
    }

    pub fn predict<C: SegmentCost>(&self, cost: &C, stop: Stop) -> Result<Segmentation, Error> {
        let min_size = effective_min_size(cost, self.grid)?;
        validate_stop(stop)?;
        let n = cost.n_samples();
        if n < min_size {
            return short_signal(n, min_size);
        }

        let mut segments = BTreeMap::new();
        segments.insert(0usize, n);
        let mut heap = BinaryHeap::new();
        if let Some(candidate) = best_split(cost, 0, n, min_size, self.grid.jump)? {
            heap.push(candidate);
        }
        let mut raw = cost.cost(0..n)?;

        loop {
            let candidate = loop {
                match heap.pop() {
                    Some(candidate) if segments.get(&candidate.start) == Some(&candidate.end) => {
                        break Some(candidate)
                    }
                    Some(_) => continue,
                    None => break None,
                }
            };
            let Some(candidate) = candidate else { break };
            if !should_split(stop, segments.len() - 1, raw, candidate.gain)? {
                break;
            }

            segments.remove(&candidate.start);
            segments.insert(candidate.start, candidate.split);
            segments.insert(candidate.split, candidate.end);
            raw -= candidate.gain;
            for (start, end) in [
                (candidate.start, candidate.split),
                (candidate.split, candidate.end),
            ] {
                if let Some(next) = best_split(cost, start, end, min_size, self.grid.jump)? {
                    heap.push(next);
                }
            }
        }
        if let Stop::Changes(changes) = stop {
            if segments.len() - 1 != changes {
                return Err(Error::InfeasibleSegmentation {
                    n_samples: n,
                    changes,
                    min_size,
                    jump: self.grid.jump,
                });
            }
        }
        finish(cost, segments.values().copied().collect(), min_size, stop)
    }
}

impl<C: SegmentCost> Detector<C> for Binseg {
    fn capabilities(&self) -> DetectorCapabilities {
        all_stops()
    }
    fn predict(&self, cost: &C, stop: Stop) -> Result<Segmentation, Error> {
        Binseg::predict(self, cost, stop)
    }
}

#[derive(Clone, Copy, Debug)]
struct SplitCandidate {
    gain: f64,
    start: usize,
    split: usize,
    end: usize,
}

impl PartialEq for SplitCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for SplitCandidate {}
impl PartialOrd for SplitCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SplitCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain
            .total_cmp(&other.gain)
            .then_with(|| other.split.cmp(&self.split))
            .then_with(|| other.start.cmp(&self.start))
            .then_with(|| other.end.cmp(&self.end))
    }
}

fn best_split<C: SegmentCost>(
    cost: &C,
    start: usize,
    end: usize,
    min_size: usize,
    jump: usize,
) -> Result<Option<SplitCandidate>, Error> {
    let parent = cost.cost(start..end)?;
    let mut best: Option<SplitCandidate> = None;
    let mut split = start.saturating_add(jump - start % jump);
    if split == start {
        split = split.saturating_add(jump);
    }
    while split < end {
        if split - start >= min_size && end - split >= min_size {
            let gain = parent - cost.cost(start..split)? - cost.cost(split..end)?;
            let candidate = SplitCandidate {
                gain,
                start,
                split,
                end,
            };
            if best.is_none_or(|current| candidate > current) {
                best = Some(candidate);
            }
        }
        split = match split.checked_add(jump) {
            Some(value) => value,
            None => break,
        };
    }
    Ok(best)
}

impl BottomUp {
    pub fn new(min_size: usize, jump: usize) -> Result<Self, Error> {
        Ok(Self {
            grid: SearchGrid::new(min_size, jump)?,
        })
    }
    pub fn grid(self) -> SearchGrid {
        self.grid
    }

    pub fn predict<C: SegmentCost>(&self, cost: &C, stop: Stop) -> Result<Segmentation, Error> {
        let min_size = effective_min_size(cost, self.grid)?;
        validate_stop(stop)?;
        let n = cost.n_samples();
        if n < min_size {
            return short_signal(n, min_size);
        }
        let leaves = initial_leaves(n, min_size, self.grid.jump);
        let mut nodes = Vec::with_capacity(leaves.len());
        for (index, &(start, end)) in leaves.iter().enumerate() {
            nodes.push(MergeNode {
                start,
                end,
                cost: cost.cost(start..end)?,
                prev: index.checked_sub(1),
                next: (index + 1 < leaves.len()).then_some(index + 1),
                generation: 0,
                active: true,
            });
        }
        let mut heap = BinaryHeap::new();
        for left in 0..nodes.len().saturating_sub(1) {
            push_merge(cost, &nodes, left, &mut heap)?;
        }
        let mut segment_count = nodes.len();
        let mut raw = nodes.iter().map(|node| node.cost).sum::<f64>();

        while let Some(candidate) = pop_valid_merge(&nodes, &mut heap) {
            if !should_merge(stop, segment_count - 1, raw, candidate.delta)? {
                break;
            }
            let left = candidate.left;
            let right = candidate.right;
            let right_next = nodes[right].next;
            nodes[left].end = nodes[right].end;
            nodes[left].cost += candidate.delta + nodes[right].cost;
            nodes[left].next = right_next;
            nodes[left].generation += 1;
            nodes[right].active = false;
            nodes[right].generation += 1;
            if let Some(next) = right_next {
                nodes[next].prev = Some(left);
            }
            raw += candidate.delta;
            segment_count -= 1;
            if let Some(prev) = nodes[left].prev {
                push_merge(cost, &nodes, prev, &mut heap)?;
            }
            push_merge(cost, &nodes, left, &mut heap)?;
        }
        if let Stop::Changes(changes) = stop {
            if segment_count - 1 != changes {
                return Err(Error::InfeasibleSegmentation {
                    n_samples: n,
                    changes,
                    min_size,
                    jump: self.grid.jump,
                });
            }
        }
        let mut breakpoints: Vec<_> = nodes
            .iter()
            .filter(|node| node.active)
            .map(|node| node.end)
            .collect();
        breakpoints.sort_unstable();
        finish(cost, breakpoints, min_size, stop)
    }
}

impl<C: SegmentCost> Detector<C> for BottomUp {
    fn capabilities(&self) -> DetectorCapabilities {
        all_stops()
    }
    fn predict(&self, cost: &C, stop: Stop) -> Result<Segmentation, Error> {
        BottomUp::predict(self, cost, stop)
    }
}

#[derive(Clone, Debug)]
struct MergeNode {
    start: usize,
    end: usize,
    cost: f64,
    prev: Option<usize>,
    next: Option<usize>,
    generation: u64,
    active: bool,
}

#[derive(Clone, Copy, Debug)]
struct MergeCandidate {
    delta: f64,
    left: usize,
    right: usize,
    left_generation: u64,
    right_generation: u64,
}
impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for MergeCandidate {}
impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .delta
            .total_cmp(&self.delta)
            .then_with(|| other.left.cmp(&self.left))
    }
}

fn initial_leaves(n: usize, min_size: usize, jump: usize) -> Vec<(usize, usize)> {
    let mut stack = vec![(0, n)];
    let mut leaves = Vec::new();
    while let Some((start, end)) = stack.pop() {
        if let Some(split) = nearest_split(start, end, min_size, jump) {
            stack.push((split, end));
            stack.push((start, split));
        } else {
            leaves.push((start, end));
        }
    }
    leaves.sort_unstable();
    leaves
}

fn nearest_split(start: usize, end: usize, min_size: usize, jump: usize) -> Option<usize> {
    let midpoint = (start + end) as f64 / 2.0;
    let mut best: Option<(f64, usize)> = None;
    let mut split = start.saturating_add(jump - start % jump);
    if split == start {
        split = split.saturating_add(jump);
    }
    while split < end {
        if split > start && split - start >= min_size && end - split >= min_size {
            let key = ((split as f64 - midpoint).abs(), split);
            if best.is_none_or(|current| key < current) {
                best = Some(key);
            }
        }
        split = split.checked_add(jump)?;
    }
    best.map(|(_, split)| split)
}

fn push_merge<C: SegmentCost>(
    cost: &C,
    nodes: &[MergeNode],
    left: usize,
    heap: &mut BinaryHeap<MergeCandidate>,
) -> Result<(), Error> {
    if !nodes[left].active {
        return Ok(());
    }
    let Some(right) = nodes[left].next else {
        return Ok(());
    };
    if !nodes[right].active {
        return Ok(());
    }
    let merged = cost.cost(nodes[left].start..nodes[right].end)?;
    heap.push(MergeCandidate {
        delta: merged - nodes[left].cost - nodes[right].cost,
        left,
        right,
        left_generation: nodes[left].generation,
        right_generation: nodes[right].generation,
    });
    Ok(())
}

fn pop_valid_merge(
    nodes: &[MergeNode],
    heap: &mut BinaryHeap<MergeCandidate>,
) -> Option<MergeCandidate> {
    while let Some(candidate) = heap.pop() {
        let left = &nodes[candidate.left];
        let right = &nodes[candidate.right];
        if left.active
            && right.active
            && left.next == Some(candidate.right)
            && left.generation == candidate.left_generation
            && right.generation == candidate.right_generation
        {
            return Some(candidate);
        }
    }
    None
}

impl Window {
    pub fn new(width: usize, min_size: usize, jump: usize) -> Result<Self, Error> {
        let grid = SearchGrid::new(min_size, jump)?;
        let width = 2 * (width / 2);
        if width < 2 * min_size {
            return Err(Error::InvalidRange {
                start: 0,
                end: width,
                n_samples: 2 * min_size,
            });
        }
        Ok(Self { grid, width })
    }
    pub fn grid(self) -> SearchGrid {
        self.grid
    }
    pub fn width(self) -> usize {
        self.width
    }

    pub fn predict<C: SegmentCost>(&self, cost: &C, stop: Stop) -> Result<Segmentation, Error> {
        let min_size = effective_min_size(cost, self.grid)?;
        validate_stop(stop)?;
        let n = cost.n_samples();
        if n < min_size {
            return short_signal(n, min_size);
        }
        let half = self.width / 2;
        let mut centers = Vec::new();
        let mut scores = Vec::new();
        let mut center = 0usize;
        while center < n {
            if center >= half && center + half <= n {
                let start = center - half;
                let end = center + half;
                scores.push(
                    cost.cost(start..end)? - cost.cost(start..center)? - cost.cost(center..end)?,
                );
                centers.push(center);
            }
            center = match center.checked_add(self.grid.jump) {
                Some(value) => value,
                None => break,
            };
        }
        let order = (self.width.max(2 * min_size) / (2 * self.grid.jump)).max(1);
        let mut peaks = Vec::new();
        for index in 0..scores.len() {
            let from = index.saturating_sub(order);
            let to = (index + order + 1).min(scores.len());
            if (from..to).all(|other| other == index || scores[index] > scores[other]) {
                peaks.push(Peak {
                    score: scores[index],
                    position: centers[index],
                });
            }
        }
        peaks.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.position.cmp(&right.position))
        });
        let mut breakpoints = vec![n];
        for peak in peaks {
            let insertion = breakpoints.partition_point(|&value| value < peak.position);
            let start = if insertion == 0 {
                0
            } else {
                breakpoints[insertion - 1]
            };
            let end = breakpoints[insertion];
            if peak.position - start < min_size || end - peak.position < min_size {
                continue;
            }
            let gain = cost.cost(start..end)?
                - cost.cost(start..peak.position)?
                - cost.cost(peak.position..end)?;
            let raw = raw_cost(cost, &breakpoints)?;
            if !should_split(stop, breakpoints.len() - 1, raw, gain)? {
                break;
            }
            breakpoints.insert(insertion, peak.position);
        }
        if let Stop::Changes(changes) = stop {
            if breakpoints.len() - 1 != changes {
                return Err(Error::InfeasibleSegmentation {
                    n_samples: n,
                    changes,
                    min_size,
                    jump: self.grid.jump,
                });
            }
        }
        finish(cost, breakpoints, min_size, stop)
    }
}

impl<C: SegmentCost> Detector<C> for Window {
    fn capabilities(&self) -> DetectorCapabilities {
        all_stops()
    }
    fn predict(&self, cost: &C, stop: Stop) -> Result<Segmentation, Error> {
        Window::predict(self, cost, stop)
    }
}

#[derive(Clone, Copy)]
struct Peak {
    score: f64,
    position: usize,
}

fn effective_min_size<C: SegmentCost>(cost: &C, grid: SearchGrid) -> Result<usize, Error> {
    if grid.min_size < cost.min_size() {
        return Err(Error::MinSizeBelowCost {
            requested: grid.min_size,
            minimum: cost.min_size(),
        });
    }
    Ok(grid.min_size)
}

fn short_signal<T>(n: usize, min_size: usize) -> Result<T, Error> {
    Err(Error::SegmentTooShort {
        start: 0,
        end: n,
        length: n,
        minimum: min_size,
    })
}

fn validate_stop(stop: Stop) -> Result<(), Error> {
    match stop {
        Stop::Penalty(value) => validate_penalty(value),
        Stop::Budget(value) => validate_budget(value),
        Stop::Changes(_) => Ok(()),
    }
}

fn all_stops() -> DetectorCapabilities {
    DetectorCapabilities {
        changes: true,
        penalty: true,
        budget: true,
    }
}

fn should_split(stop: Stop, changes: usize, raw: f64, gain: f64) -> Result<bool, Error> {
    Ok(match stop {
        Stop::Changes(target) => changes < target,
        Stop::Penalty(penalty) => gain > penalty && !objective_values_tied(gain, penalty),
        Stop::Budget(budget) => raw > budget && !objective_values_tied(raw, budget),
    })
}

fn should_merge(stop: Stop, changes: usize, raw: f64, delta: f64) -> Result<bool, Error> {
    Ok(match stop {
        Stop::Changes(target) => changes > target,
        Stop::Penalty(penalty) => delta < penalty && !objective_values_tied(delta, penalty),
        Stop::Budget(budget) => raw + delta <= budget || objective_values_tied(raw + delta, budget),
    })
}

fn raw_cost<C: SegmentCost>(cost: &C, breakpoints: &[usize]) -> Result<f64, Error> {
    let mut start = 0;
    let mut total = 0.0;
    for &end in breakpoints {
        total += cost.cost(start..end)?;
        start = end;
    }
    Ok(total)
}

fn finish<C: SegmentCost>(
    cost: &C,
    breakpoints: Vec<usize>,
    min_size: usize,
    stop: Stop,
) -> Result<Segmentation, Error> {
    let raw = raw_cost(cost, &breakpoints)?;
    let objective = match stop {
        Stop::Penalty(penalty) => raw + penalty * (breakpoints.len() - 1) as f64,
        _ => raw,
    };
    Segmentation::new(breakpoints, raw, objective, cost.n_samples(), min_size)
}

#[cfg(test)]
#[path = "../../tests/unit/search/greedy.rs"]
mod tests;
