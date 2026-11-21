// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The neighbourhoods on a schedule held as plain lists, the reference the tests of the scans check against: the
//! moves each proposes, in its order, written from their definitions with positions on lists, and the hand-built
//! instance the tests run on.

use super::scan::{VesselMoves, VesselScan};
use crate::schedule::{Mutator, ScheduleGraph};
use talos_core::ClosedOpenInterval;
use talos_model::{BerthIndex, Model, Solution, VesselIndex};

/// The vessels of each berth in order.
pub(super) type Order = Vec<Vec<usize>>;

/// The vessels of the instance.
const NUM_VESSELS: usize = 7;

/// The berths of the instance.
const NUM_BERTHS: usize = 5;

/// The start: B0 holds V0 to V4 in order, B1 and B3 one vessel each, B2 and B4 are empty.
const START: [&[usize]; NUM_BERTHS] = [&[0, 1, 2, 3, 4], &[5], &[], &[6], &[]];

/// The pairs of a vessel and a berth it may not use: V3 may not use B2, V5 not B0 and V6 not B4.
const FORBIDDEN: [(usize, usize); 3] = [(3, 2), (5, 0), (6, 4)];

/// The window of each vessel from its arrival to its latest departure: that of V4 misses that of V0 alone.
pub(super) const WINDOWS: [(i64, i64); NUM_VESSELS] = [
    (0, 400),
    (0, 1000),
    (0, 1000),
    (0, 1000),
    (500, 600),
    (0, 1000),
    (0, 1000),
];

/// Windows that all overlap, wide enough that every order of allowed berths is feasible.
pub(super) const OVERLAPPING_WINDOWS: [(i64, i64); NUM_VESSELS] = [(0, 1000); NUM_VESSELS];

/// The instance of `windows`: every vessel takes 10 at every berth it may use, all berths are open on [0, 1000).
pub(super) fn model(windows: &[(i64, i64); NUM_VESSELS]) -> Model<i64> {
    let mut processing = vec![Some(10); NUM_VESSELS * NUM_BERTHS];
    for (vessel, berth) in FORBIDDEN {
        processing[vessel * NUM_BERTHS + berth] = None;
    }
    Model::new(
        windows.map(|(arrival, _)| arrival).to_vec(),
        windows.map(|(_, departure)| departure).to_vec(),
        vec![1; NUM_VESSELS],
        processing,
        vec![vec![ClosedOpenInterval::new(0, 1000)]; NUM_BERTHS],
    )
}

/// The graph of the start.
pub(super) fn start_graph() -> ScheduleGraph {
    graph_of(&start())
}

/// The graph of `order`, which holds every vessel once on one of the instance's berths.
pub(super) fn graph_of(order: &Order) -> ScheduleGraph {
    debug_assert!(
        order.len() == NUM_BERTHS && order.iter().map(Vec::len).sum::<usize>() == NUM_VESSELS,
        "called `graph_of` with {order:?}, not an order of {NUM_VESSELS} vessels on {NUM_BERTHS} berths"
    );
    let mut berths = vec![BerthIndex::new(0); NUM_VESSELS];
    let mut starts = vec![0_i64; NUM_VESSELS];
    for (berth, vessels) in order.iter().enumerate() {
        for (place, &vessel) in vessels.iter().enumerate() {
            berths[vessel] = BerthIndex::new(berth);
            starts[vessel] = i64::try_from(place).expect("a place should fit an `i64`");
        }
    }
    ScheduleGraph::from_slices(&berths, &starts, NUM_BERTHS)
}

/// The order of the start.
pub(super) fn start() -> Order {
    START.iter().map(|vessels| vessels.to_vec()).collect()
}

/// The vessels of each berth of `graph` in order.
pub(super) fn order_of(graph: &ScheduleGraph) -> Order {
    graph
        .berths()
        .map(|berth| {
            graph
                .vessel_sequence_iter(berth)
                .map(VesselIndex::index)
                .collect()
        })
        .collect()
}

/// Whether `vessel` may use `berth`.
fn allowed(vessel: usize, berth: usize) -> bool {
    !FORBIDDEN.contains(&(vessel, berth))
}

/// Whether the windows of `a` and `b` overlap.
fn overlap(windows: &[(i64, i64); NUM_VESSELS], a: usize, b: usize) -> bool {
    windows[a].0 < windows[b].1 && windows[b].0 < windows[a].1
}

/// The berth of `vessel` in `order` and its position there.
fn locate(order: &Order, vessel: usize) -> (usize, usize) {
    order
        .iter()
        .enumerate()
        .find_map(|(berth, vessels)| {
            vessels
                .iter()
                .position(|&other| other == vessel)
                .map(|position| (berth, position))
        })
        .expect("every vessel should stand on one berth")
}

/// `order` with `a` and `b` exchanged.
pub(super) fn swapped(order: &Order, a: usize, b: usize) -> Order {
    let ((berth_a, position_a), (berth_b, position_b)) = (locate(order, a), locate(order, b));
    let mut order = order.clone();
    order[berth_a][position_a] = b;
    order[berth_b][position_b] = a;
    order
}

/// `order` with `vessel` taken out and inserted at `position` of `berth`, counted after it is taken out.
pub(super) fn inserted(order: &Order, vessel: usize, berth: usize, position: usize) -> Order {
    let (from, at) = locate(order, vessel);
    let mut order = order.clone();
    order[from].remove(at);
    order[berth].insert(position, vessel);
    order
}

/// The swaps of each vessel, in increasing order, with each vessel of higher index on its berth whose window
/// overlaps its own, in increasing order.
pub(super) fn intra_berth_swaps(windows: &[(i64, i64); NUM_VESSELS]) -> Vec<Order> {
    let start = start();
    let mut moves = Vec::new();
    for vessel in 0..NUM_VESSELS {
        for partner in vessel + 1..NUM_VESSELS {
            if locate(&start, partner).0 == locate(&start, vessel).0
                && overlap(windows, vessel, partner)
            {
                moves.push(swapped(&start, vessel, partner));
            }
        }
    }
    moves
}

/// The swaps of each vessel, in increasing order, with each vessel of higher index on another berth, in increasing
/// order, where each may use the other's berth.
pub(super) fn inter_berth_swaps() -> Vec<Order> {
    let start = start();
    let mut moves = Vec::new();
    for vessel in 0..NUM_VESSELS {
        let (own, _) = locate(&start, vessel);
        for partner in vessel + 1..NUM_VESSELS {
            let (berth, _) = locate(&start, partner);
            if berth != own && allowed(vessel, berth) && allowed(partner, own) {
                moves.push(swapped(&start, vessel, partner));
            }
        }
    }
    moves
}

/// The insertions of each vessel, in increasing order, on its own berth at least two places from where it stands,
/// next to a vessel whose window overlaps its own: after each vessel in increasing order of that vessel, the one
/// it would follow, then at the head, before the vessel it would precede.
pub(super) fn intra_berth_shifts(windows: &[(i64, i64); NUM_VESSELS]) -> Vec<Order> {
    let start = start();
    let mut moves = Vec::new();
    for vessel in 0..NUM_VESSELS {
        let (berth, position) = locate(&start, vessel);
        let vessels = &start[berth];
        for anchor in 0..NUM_VESSELS {
            let Some(after) = vessels.iter().position(|&other| other == anchor) else {
                continue;
            };
            // The place after the vessel at `after` is the one before position `after + 1`; the vessel moves from
            // `position` to that place's position once it is taken out.
            let far = after + 2 < position || after > position + 1;
            if far && overlap(windows, vessel, anchor) {
                let target = if after < position { after + 1 } else { after };
                moves.push(inserted(&start, vessel, berth, target));
            }
        }
        if position >= 2 && overlap(windows, vessel, vessels[0]) {
            moves.push(inserted(&start, vessel, berth, 0));
        }
    }
    moves
}

/// The insertions of each vessel, in increasing order, on each other berth it may use: after each vessel of such a
/// berth in increasing order of that vessel, then at the head of each such berth in increasing order of the berth.
pub(super) fn inter_berth_shifts() -> Vec<Order> {
    let start = start();
    let mut moves = Vec::new();
    for vessel in 0..NUM_VESSELS {
        let (own, _) = locate(&start, vessel);
        for anchor in 0..NUM_VESSELS {
            let (berth, after) = locate(&start, anchor);
            if berth != own && allowed(vessel, berth) {
                moves.push(inserted(&start, vessel, berth, after + 1));
            }
        }
        for berth in (0..NUM_BERTHS).filter(|&berth| berth != own && allowed(vessel, berth)) {
            moves.push(inserted(&start, vessel, berth, 0));
        }
    }
    moves
}

/// Every order other than `start` that one swap of two vessels or one insertion of a vessel at the head of a
/// berth makes from `start`, each once, in increasing order.
pub(super) fn swap_or_head_neighbors(start: &Order) -> Vec<Order> {
    let mut orders = Vec::new();
    for vessel in 0..NUM_VESSELS {
        for partner in vessel + 1..NUM_VESSELS {
            orders.push(swapped(start, vessel, partner));
        }
        for berth in 0..NUM_BERTHS {
            orders.push(inserted(start, vessel, berth, 0));
        }
    }
    distinct_neighbors(orders, start)
}

/// Every order other than `start` that one swap of two vessels or one insertion of a vessel anywhere makes from
/// `start`, each once, in increasing order.
pub(super) fn neighbors(start: &Order) -> Vec<Order> {
    let mut orders = Vec::new();
    for vessel in 0..NUM_VESSELS {
        for partner in vessel + 1..NUM_VESSELS {
            orders.push(swapped(start, vessel, partner));
        }
        let (own, _) = locate(start, vessel);
        for berth in 0..NUM_BERTHS {
            let places = start[berth].len() + usize::from(berth != own);
            for position in 0..places {
                orders.push(inserted(start, vessel, berth, position));
            }
        }
    }
    distinct_neighbors(orders, start)
}

/// `orders` sorted, each once, without `start`.
fn distinct_neighbors(mut orders: Vec<Order>, start: &Order) -> Vec<Order> {
    orders.sort();
    orders.dedup();
    orders.retain(|order| order != start);
    orders
}

/// The orders of `orders` that the plain decoder finds feasible under `model`, in their order.
pub(super) fn feasible(mut orders: Vec<Order>, model: &Model<i64>) -> Vec<Order> {
    orders.retain(|order| is_feasible(order, model));
    orders
}

/// Whether `order` is feasible: each berth served in its order, every vessel from the later of its arrival and
/// the end of the service before it, checked by the model.
pub(super) fn is_feasible(order: &Order, model: &Model<i64>) -> bool {
    let mut berths = vec![BerthIndex::new(0); NUM_VESSELS];
    let mut starts = vec![0_i64; NUM_VESSELS];
    for (berth, vessels) in order.iter().enumerate() {
        let mut free_from = 0;
        for &vessel in vessels {
            let id = VesselIndex::new(vessel);
            let Some(processing) = model.vessel_processing_time(id, BerthIndex::new(berth)) else {
                return false;
            };
            let begin = free_from.max(model.vessel_arrival_time(id));
            berths[vessel] = BerthIndex::new(berth);
            starts[vessel] = begin;
            free_from = begin + processing;
        }
    }
    // The check reads no objective value.
    let schedule = Solution::new(berths, starts, 0);
    model.check_feasibility(schedule.as_view()).is_ok()
}

/// Every move of a scan of `K` on the start under `model`, as the order each makes; each is undone before the
/// next, as a run undoes a rejected candidate.
pub(super) fn moves_of<K>(model: &Model<i64>) -> Vec<Order>
where
    K: VesselMoves,
{
    let mut scan = VesselScan::<K>::new();
    scan.prepare(&start_graph());
    let mut mutator = Mutator::new(start_graph());
    let mut moves = Vec::new();
    // No neighbourhood holds more than a move per vessel and per place, 7 * (7 + 5) of them.
    for _ in 0..=84 {
        // SAFETY: the mutator's graph, rolled back after each move, has the counts the scan was prepared for,
        // which are the model's.
        if !unsafe { scan.next_neighbor_unchecked(model, &mut mutator) } {
            return moves;
        }
        moves.push(order_of(mutator.graph()));
        mutator.rollback();
    }
    panic!("the scan should end within its 84 placements")
}
