// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A problem instance: the vessels, the berths, and which vessel may use which berth for how long.

use crate::index::{BerthIndex, BerthIndexIter, VesselIndex, VesselIndexIter};
use std::fmt;
use talos_core::{ClosedOpenInterval, SolverNumeric};
use typed_index_collections::TiVec;

/// An instance of the dynamic berth allocation problem.
///
/// Each vessel arrives at a time, must have left by its latest departure time, and weighs its time in port by a
/// weight. Each pair of a vessel and a berth has a processing time, or none where the vessel may not use the
/// berth. Each berth is open during a set of intervals, and a vessel's service must lie within one of them.
///
/// A model is cold data: it is built once, checked as it is built, and read by the solver. Every accessor
/// answers in constant time. The accessors indexed by a vessel or a berth come as a checked form, which panics on
/// an id beyond the model, and an `_unchecked` form for a caller whose loop proves the id.
///
/// # Examples
///
/// ```rust
/// use talos_core::ClosedOpenInterval;
/// use talos_model::{BerthIndex, Model, VesselIndex};
///
/// // Two vessels and one berth, open from 0 to 100; the second vessel may not use the berth.
/// let model = Model::new(
///     vec![0, 10],         // arrival times
///     vec![50, 60],        // latest departure times
///     vec![1, 2],          // weights
///     vec![Some(5), None], // processing times, one per vessel and berth
///     vec![vec![ClosedOpenInterval::new(0, 100)]],
/// );
///
/// let (first, second, berth) = (VesselIndex::new(0), VesselIndex::new(1), BerthIndex::new(0));
/// assert_eq!(model.vessel_processing_time(first, berth), Some(5));
/// assert_eq!(model.vessel_processing_time(second, berth), None);
/// assert_eq!(model.berth_opening_times(berth), [ClosedOpenInterval::new(0, 100)]);
/// ```
// Invariant: the three vessel columns have one entry per vessel, and `processing_times` has one entry per pair of a
// vessel and a berth, the pair `(v, b)` at `v * num_berths() + b`. There are at most `u32::MAX` vessels and at most
// `u32::MAX` berths, as `check_counts` refuses more, so that an id names each of them and the count itself, which
// `vessel_iter` and `berth_iter` rely on to walk them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model<T>
where
    T: SolverNumeric,
{
    arrival_times: TiVec<VesselIndex, T>,
    latest_departure_times: TiVec<VesselIndex, T>,
    weights: TiVec<VesselIndex, T>,
    processing_times: Vec<ProcessingTime<T>>,
    openings: Openings<T>,
}

impl<T> Model<T>
where
    T: SolverNumeric,
{
    /// The model of the given vessels and berths, as [`try_new`](Model::try_new) makes it.
    ///
    /// # Panics
    ///
    /// Panics if `try_new` refuses the input, with the [`ModelError`] it answers.
    #[track_caller]
    pub fn new(
        arrival_times: Vec<T>,
        latest_departure_times: Vec<T>,
        weights: Vec<T>,
        processing_times: Vec<Option<T>>,
        openings: Vec<Vec<ClosedOpenInterval<T>>>,
    ) -> Self {
        match Self::try_new(
            arrival_times,
            latest_departure_times,
            weights,
            processing_times,
            openings,
        ) {
            Ok(model) => model,
            Err(error) => panic!("called `Model::new` with {error}"),
        }
    }

    /// The model of the given vessels and berths, or why they make none.
    ///
    /// The vessels are those of `arrival_times`; `latest_departure_times` and `weights` hold one entry for each
    /// of them, in the same order. The berths are those of `openings`, which holds for each berth the intervals
    /// during which it is open. `processing_times` holds one entry per pair of a vessel and a berth, the pairs of
    /// the first vessel first: the entry of vessel `v` at berth `b` is at `v * openings.len() + b`, and `None`
    /// where the vessel may not use the berth.
    ///
    /// Each berth's openings are kept as their union: the model answers them ordered by start, without empty,
    /// overlapping or touching intervals, whatever order and overlap they were given in.
    ///
    /// # Errors
    ///
    /// - [`ModelError::TooManyVessels`] or [`ModelError::TooManyBerths`] if there are more vessels or berths than
    ///   ids can name.
    /// - [`ModelError::LatestDepartureTimesLength`] or [`ModelError::WeightsLength`] if a vessel column does not
    ///   hold one entry per vessel.
    /// - [`ModelError::ProcessingTimesLength`] if `processing_times` does not hold one entry per pair.
    /// - [`ModelError::NegativeProcessingTime`] for the first processing time below zero.
    pub fn try_new(
        arrival_times: Vec<T>,
        latest_departure_times: Vec<T>,
        weights: Vec<T>,
        processing_times: Vec<Option<T>>,
        openings: Vec<Vec<ClosedOpenInterval<T>>>,
    ) -> Result<Self, ModelError<T>> {
        let num_vessels = arrival_times.len();
        let num_berths = openings.len();

        check_counts(num_vessels, num_berths)?;
        check_vessel_columns(num_vessels, &latest_departure_times, &weights)?;
        let processing_times = processing_time_column(&processing_times, num_vessels, num_berths)?;

        Ok(Self {
            arrival_times: TiVec::from(arrival_times),
            latest_departure_times: TiVec::from(latest_departure_times),
            weights: TiVec::from(weights),
            processing_times,
            openings: Openings::new(&openings),
        })
    }

    /// The number of vessels.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.arrival_times.len()
    }

    /// The number of berths.
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.openings.num_berths()
    }

    /// The ids of every vessel, in increasing order.
    #[inline]
    pub fn vessel_iter(&self) -> VesselIndexIter {
        VesselIndexIter::new(self.num_vessels())
    }

    /// The ids of every berth, in increasing order.
    #[inline]
    pub fn berth_iter(&self) -> BerthIndexIter {
        BerthIndexIter::new(self.num_berths())
    }

    /// The time `vessel` arrives.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not a vessel of the model.
    #[inline]
    #[track_caller]
    pub fn vessel_arrival_time(&self, vessel: VesselIndex) -> T {
        if vessel.index() >= self.num_vessels() {
            arrival_time_of_vessel_beyond_model(vessel, self.num_vessels());
        }
        // SAFETY: `vessel` is a vessel of the model, as the check above found in every build.
        unsafe { self.vessel_arrival_time_unchecked(vessel) }
    }

    /// The time `vessel` arrives, without checking the id.
    ///
    /// # Safety
    ///
    /// `vessel` must be a vessel of the model, as [`vessel_arrival_time`](Model::vessel_arrival_time) checks.
    #[inline]
    pub unsafe fn vessel_arrival_time_unchecked(&self, vessel: VesselIndex) -> T {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `Model::vessel_arrival_time_unchecked` with {vessel}, beyond the {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is a vessel of the model, as the caller guarantees, and `arrival_times` holds one entry
        // per vessel, as the invariant of `Model` states.
        unsafe { *self.arrival_times.get_unchecked(vessel) }
    }

    /// The time by which `vessel` must have left.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not a vessel of the model.
    #[inline]
    #[track_caller]
    pub fn vessel_latest_departure_time(&self, vessel: VesselIndex) -> T {
        if vessel.index() >= self.num_vessels() {
            latest_departure_time_of_vessel_beyond_model(vessel, self.num_vessels());
        }
        // SAFETY: `vessel` is a vessel of the model, as the check above found in every build.
        unsafe { self.vessel_latest_departure_time_unchecked(vessel) }
    }

    /// The time by which `vessel` must have left, without checking the id.
    ///
    /// # Safety
    ///
    /// `vessel` must be a vessel of the model, as
    /// [`vessel_latest_departure_time`](Model::vessel_latest_departure_time) checks.
    #[inline]
    pub unsafe fn vessel_latest_departure_time_unchecked(&self, vessel: VesselIndex) -> T {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `Model::vessel_latest_departure_time_unchecked` with {vessel}, beyond the {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is a vessel of the model, as the caller guarantees, and `latest_departure_times` holds
        // one entry per vessel, as the invariant of `Model` states.
        unsafe { *self.latest_departure_times.get_unchecked(vessel) }
    }

    /// The weight of `vessel`, by which its time in port counts.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not a vessel of the model.
    #[inline]
    #[track_caller]
    pub fn vessel_weight(&self, vessel: VesselIndex) -> T {
        if vessel.index() >= self.num_vessels() {
            weight_of_vessel_beyond_model(vessel, self.num_vessels());
        }
        // SAFETY: `vessel` is a vessel of the model, as the check above found in every build.
        unsafe { self.vessel_weight_unchecked(vessel) }
    }

    /// The weight of `vessel`, without checking the id.
    ///
    /// # Safety
    ///
    /// `vessel` must be a vessel of the model, as [`vessel_weight`](Model::vessel_weight) checks.
    #[inline]
    pub unsafe fn vessel_weight_unchecked(&self, vessel: VesselIndex) -> T {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `Model::vessel_weight_unchecked` with {vessel}, beyond the {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is a vessel of the model, as the caller guarantees, and `weights` holds one entry per
        // vessel, as the invariant of `Model` states.
        unsafe { *self.weights.get_unchecked(vessel) }
    }

    /// How long serving `vessel` at `berth` takes, or `None` if the vessel may not use the berth.
    ///
    /// A processing time is never negative.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not a vessel or `berth` not a berth of the model.
    #[inline]
    #[track_caller]
    pub fn vessel_processing_time(&self, vessel: VesselIndex, berth: BerthIndex) -> Option<T> {
        if vessel.index() >= self.num_vessels() {
            processing_time_of_vessel_beyond_model(vessel, self.num_vessels());
        }
        if berth.index() >= self.num_berths() {
            processing_time_at_berth_beyond_model(berth, self.num_berths());
        }
        // SAFETY: `vessel` is a vessel and `berth` a berth of the model, as the checks above found in every build.
        unsafe { self.vessel_processing_time_unchecked(vessel, berth) }
    }

    /// How long serving `vessel` at `berth` takes, or `None` if the vessel may not use the berth, without
    /// checking the ids.
    ///
    /// # Safety
    ///
    /// `vessel` must be a vessel and `berth` a berth of the model, as
    /// [`vessel_processing_time`](Model::vessel_processing_time) checks.
    #[inline]
    pub unsafe fn vessel_processing_time_unchecked(
        &self,
        vessel: VesselIndex,
        berth: BerthIndex,
    ) -> Option<T> {
        let entry = self.processing_entry(vessel, berth);
        debug_assert!(
            vessel.index() < self.num_vessels() && berth.index() < self.num_berths(),
            "called `Model::vessel_processing_time_unchecked` with {vessel} and {berth}, beyond the {} vessels or \
             the {} berths",
            self.num_vessels(),
            self.num_berths()
        );
        // SAFETY: `vessel` and `berth` are of the model, as the caller guarantees, and `processing_times` holds the
        // entry of every such pair at `processing_entry`, as the invariant of `Model` states.
        unsafe { self.processing_times.get_unchecked(entry) }.get()
    }

    /// The intervals during which `berth` is open: non-empty, ordered by start, and each separated from the next
    /// by a gap.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not a berth of the model.
    #[inline]
    #[track_caller]
    pub fn berth_opening_times(&self, berth: BerthIndex) -> &[ClosedOpenInterval<T>] {
        if berth.index() >= self.num_berths() {
            opening_times_of_berth_beyond_model(berth, self.num_berths());
        }
        // SAFETY: `berth` is a berth of the model, as the check above found in every build.
        unsafe { self.berth_opening_times_unchecked(berth) }
    }

    /// The intervals during which `berth` is open, without checking the id.
    ///
    /// # Safety
    ///
    /// `berth` must be a berth of the model, as [`berth_opening_times`](Model::berth_opening_times) checks.
    #[inline]
    pub unsafe fn berth_opening_times_unchecked(
        &self,
        berth: BerthIndex,
    ) -> &[ClosedOpenInterval<T>] {
        debug_assert!(
            berth.index() < self.num_berths(),
            "called `Model::berth_opening_times_unchecked` with {berth}, beyond the {} berths",
            self.num_berths()
        );
        // SAFETY: `berth` is a berth of the model, as the caller guarantees.
        unsafe { self.openings.of_berth_unchecked(berth) }
    }

    /// The position of the pair of `vessel` and `berth` in `processing_times`.
    #[inline]
    fn processing_entry(&self, vessel: VesselIndex, berth: BerthIndex) -> usize {
        vessel.index() * self.num_berths() + berth.index()
    }
}

// The panics of the checked accessors stay out of line, so that an inlined accessor is a compare and a read. Each
// is `#[cold]` because a caller reaches it only by a mistake, `#[inline(never)]` so that its formatting is never
// copied into a caller, and `#[track_caller]` so that the panic names the caller's line.

/// Panics for [`Model::vessel_arrival_time`] with a vessel beyond the model's `num_vessels`.
#[cold]
#[inline(never)]
#[track_caller]
fn arrival_time_of_vessel_beyond_model(vessel: VesselIndex, num_vessels: usize) -> ! {
    panic!("called `Model::vessel_arrival_time` with {vessel}, beyond the {num_vessels} vessels")
}

/// Panics for [`Model::vessel_latest_departure_time`] with a vessel beyond the model's `num_vessels`.
#[cold]
#[inline(never)]
#[track_caller]
fn latest_departure_time_of_vessel_beyond_model(vessel: VesselIndex, num_vessels: usize) -> ! {
    panic!(
        "called `Model::vessel_latest_departure_time` with {vessel}, beyond the {num_vessels} vessels"
    )
}

/// Panics for [`Model::vessel_weight`] with a vessel beyond the model's `num_vessels`.
#[cold]
#[inline(never)]
#[track_caller]
fn weight_of_vessel_beyond_model(vessel: VesselIndex, num_vessels: usize) -> ! {
    panic!("called `Model::vessel_weight` with {vessel}, beyond the {num_vessels} vessels")
}

/// Panics for [`Model::vessel_processing_time`] with a vessel beyond the model's `num_vessels`.
#[cold]
#[inline(never)]
#[track_caller]
fn processing_time_of_vessel_beyond_model(vessel: VesselIndex, num_vessels: usize) -> ! {
    panic!("called `Model::vessel_processing_time` with {vessel}, beyond the {num_vessels} vessels")
}

/// Panics for [`Model::vessel_processing_time`] with a berth beyond the model's `num_berths`.
#[cold]
#[inline(never)]
#[track_caller]
fn processing_time_at_berth_beyond_model(berth: BerthIndex, num_berths: usize) -> ! {
    panic!("called `Model::vessel_processing_time` with {berth}, beyond the {num_berths} berths")
}

/// Panics for [`Model::berth_opening_times`] with a berth beyond the model's `num_berths`.
#[cold]
#[inline(never)]
#[track_caller]
fn opening_times_of_berth_beyond_model(berth: BerthIndex, num_berths: usize) -> ! {
    panic!("called `Model::berth_opening_times` with {berth}, beyond the {num_berths} berths")
}

/// Refuses more vessels or berths than ids can name.
///
/// # Errors
///
/// [`ModelError::TooManyVessels`] or [`ModelError::TooManyBerths`].
fn check_counts<T>(num_vessels: usize, num_berths: usize) -> Result<(), ModelError<T>> {
    if !ids_can_name(num_vessels) {
        return Err(ModelError::TooManyVessels { count: num_vessels });
    }
    if !ids_can_name(num_berths) {
        return Err(ModelError::TooManyBerths { count: num_berths });
    }
    Ok(())
}

/// Whether an id names each of `count` positions and the count itself, which ends a walk over them.
fn ids_can_name(count: usize) -> bool {
    u32::try_from(count).is_ok()
}

/// Refuses a vessel column without one entry per vessel.
///
/// # Errors
///
/// [`ModelError::LatestDepartureTimesLength`] or [`ModelError::WeightsLength`].
fn check_vessel_columns<T>(
    num_vessels: usize,
    latest_departure_times: &[T],
    weights: &[T],
) -> Result<(), ModelError<T>> {
    if latest_departure_times.len() != num_vessels {
        return Err(ModelError::LatestDepartureTimesLength {
            len: latest_departure_times.len(),
            num_vessels,
        });
    }
    if weights.len() != num_vessels {
        return Err(ModelError::WeightsLength {
            len: weights.len(),
            num_vessels,
        });
    }
    Ok(())
}

/// The stored processing times of every pair of a vessel and a berth, the pairs of the first vessel first.
///
/// # Errors
///
/// [`ModelError::ProcessingTimesLength`] if there is not one entry per pair, and
/// [`ModelError::NegativeProcessingTime`] for the first time below zero.
fn processing_time_column<T>(
    processing_times: &[Option<T>],
    num_vessels: usize,
    num_berths: usize,
) -> Result<Vec<ProcessingTime<T>>, ModelError<T>>
where
    T: SolverNumeric,
{
    // No vector holds more entries than a `usize` counts, so a product beyond it matches no length.
    if num_vessels.checked_mul(num_berths) != Some(processing_times.len()) {
        return Err(ModelError::ProcessingTimesLength {
            len: processing_times.len(),
            num_vessels,
            num_berths,
        });
    }

    // An entry exists only where the product, and so `num_berths`, is positive, so the division is defined; the
    // quotient and the remainder are below the counts, which ids can name.
    processing_times
        .iter()
        .enumerate()
        .map(|(entry, &time)| match time {
            Some(processing_time) if processing_time < T::ZERO => {
                Err(ModelError::NegativeProcessingTime {
                    vessel: VesselIndex::new(entry / num_berths),
                    berth: BerthIndex::new(entry % num_berths),
                    processing_time,
                })
            }
            _ => Ok(ProcessingTime::new(time)),
        })
        .collect()
}

/// A processing time, or its absence where a vessel may not use a berth, stored in one word of `T`.
///
/// A present time is never negative, so a negative word can mean absence.
// Invariant: the word is a present time, at least zero, or `T::NEGATIVE_ONE` for absence.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ProcessingTime<T>(T);

impl<T> ProcessingTime<T>
where
    T: SolverNumeric,
{
    /// `time` in its stored form; a present `time` is at least zero.
    #[inline]
    fn new(time: Option<T>) -> Self {
        match time {
            Some(time) => {
                debug_assert!(
                    time >= T::ZERO,
                    "called `ProcessingTime::new` with negative time {time}"
                );
                Self(time)
            }
            None => Self(T::NEGATIVE_ONE),
        }
    }

    /// The time, or `None` where it is absent.
    #[inline]
    fn get(self) -> Option<T> {
        (self.0 >= T::ZERO).then_some(self.0)
    }
}

// The stored word would show absence as -1; the debug form shows the option it stands for.
impl<T> fmt::Debug for ProcessingTime<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.get(), f)
    }
}

/// The intervals during which each berth is open, all berths' in one column.
// Invariant: `offsets` has one entry per berth and one more; it starts at 0, never decreases and ends at
// `intervals.len()`, so the openings of berth `b` are `intervals[offsets[b]..offsets[b + 1]]`. The openings of each
// berth are non-empty, ordered by start and separated by gaps, as `union_of_openings` makes them.
//
// An offset is a position in `intervals`, a count of the openings before a berth's, and so a `usize`: no id names
// an opening, so no `u32` bound is held for them, and the length of `intervals` is bounded only as any `Vec`'s is.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Openings<T>
where
    T: SolverNumeric,
{
    intervals: Vec<ClosedOpenInterval<T>>,
    offsets: Vec<usize>,
}

impl<T> Openings<T>
where
    T: SolverNumeric,
{
    /// The openings of the berths of `per_berth`, each berth's as the union of the intervals given for it.
    fn new(per_berth: &[Vec<ClosedOpenInterval<T>>]) -> Self {
        let mut intervals = Vec::new();
        let mut offsets = Vec::with_capacity(per_berth.len() + 1);
        offsets.push(0);
        for openings in per_berth {
            intervals.extend(union_of_openings(openings));
            offsets.push(intervals.len());
        }
        Self { intervals, offsets }
    }

    /// The number of berths.
    #[inline]
    fn num_berths(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The openings of `berth`, without checking the id.
    ///
    /// # Safety
    ///
    /// `berth` must be one of the berths, below [`num_berths`](Openings::num_berths).
    #[inline]
    unsafe fn of_berth_unchecked(&self, berth: BerthIndex) -> &[ClosedOpenInterval<T>] {
        debug_assert!(
            berth.index() < self.num_berths()
                && self.offsets[berth.index()] <= self.offsets[berth.index() + 1]
                && self.offsets[berth.index() + 1] <= self.intervals.len(),
            "called `Openings::of_berth_unchecked` with {berth}, beyond the {} berths or without ordered offsets \
             within the {} intervals",
            self.num_berths(),
            self.intervals.len()
        );
        // SAFETY: `berth` is below the number of berths, as the caller guarantees, so `offsets` holds the entries at
        // it and after it, and they bound a range of `intervals`, as the invariant of `Openings` states.
        unsafe {
            let (start, end) = (
                *self.offsets.get_unchecked(berth.index()),
                *self.offsets.get_unchecked(berth.index() + 1),
            );
            self.intervals.get_unchecked(start..end)
        }
    }
}

/// The union of `openings` as the fewest intervals: non-empty, ordered by start, and each separated from the next
/// by a gap of at least one integer.
fn union_of_openings<T>(openings: &[ClosedOpenInterval<T>]) -> Vec<ClosedOpenInterval<T>>
where
    T: SolverNumeric,
{
    let mut ordered: Vec<ClosedOpenInterval<T>> = openings
        .iter()
        .copied()
        .filter(|opening| !opening.is_empty())
        .collect();
    ordered.sort_unstable_by_key(ClosedOpenInterval::start);

    let mut union: Vec<ClosedOpenInterval<T>> = Vec::with_capacity(ordered.len());
    for opening in ordered {
        // `last` starts no later than `opening`, so the two join exactly when `opening` starts by the end of `last`.
        if let Some(last) = union.last_mut()
            && let Some(joined) = last.union(opening)
        {
            *last = joined;
        } else {
            union.push(opening);
        }
    }
    union
}

/// Why vessels and berths make no [`Model`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelError<T> {
    /// There are more vessels than ids can name.
    TooManyVessels {
        /// The number of vessels given.
        count: usize,
    },
    /// There are more berths than ids can name.
    TooManyBerths {
        /// The number of berths given.
        count: usize,
    },
    /// The latest departure times are not one per vessel.
    LatestDepartureTimesLength {
        /// The number of latest departure times given.
        len: usize,
        /// The number of vessels, that of the arrival times.
        num_vessels: usize,
    },
    /// The weights are not one per vessel.
    WeightsLength {
        /// The number of weights given.
        len: usize,
        /// The number of vessels, that of the arrival times.
        num_vessels: usize,
    },
    /// The processing times are not one per pair of a vessel and a berth.
    ProcessingTimesLength {
        /// The number of processing times given.
        len: usize,
        /// The number of vessels, that of the arrival times.
        num_vessels: usize,
        /// The number of berths, that of the openings.
        num_berths: usize,
    },
    /// A processing time is below zero.
    NegativeProcessingTime {
        /// The vessel of the pair.
        vessel: VesselIndex,
        /// The berth of the pair.
        berth: BerthIndex,
        /// The processing time given.
        processing_time: T,
    },
}

impl<T> fmt::Display for ModelError<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyVessels { count } => {
                write!(
                    f,
                    "{count} vessels, more than the {} that ids can name",
                    u32::MAX
                )
            }
            Self::TooManyBerths { count } => {
                write!(
                    f,
                    "{count} berths, more than the {} that ids can name",
                    u32::MAX
                )
            }
            Self::LatestDepartureTimesLength { len, num_vessels } => {
                write!(f, "{len} latest departure times for {num_vessels} vessels")
            }
            Self::WeightsLength { len, num_vessels } => {
                write!(f, "{len} weights for {num_vessels} vessels")
            }
            Self::ProcessingTimesLength {
                len,
                num_vessels,
                num_berths,
            } => write!(
                f,
                "{len} processing times for {num_vessels} vessels and {num_berths} berths, which need one per pair"
            ),
            Self::NegativeProcessingTime {
                vessel,
                berth,
                processing_time,
            } => write!(
                f,
                "negative processing time {processing_time} of {vessel} at {berth}"
            ),
        }
    }
}

impl<T> std::error::Error for ModelError<T> where T: fmt::Debug + fmt::Display {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn interval(start: i64, end: i64) -> ClosedOpenInterval<i64> {
        ClosedOpenInterval::new(start, end)
    }

    /// Two vessels and two berths; the first vessel may not use the second berth.
    fn two_by_two() -> Model<i64> {
        Model::new(
            vec![0, 10],
            vec![100, 110],
            vec![1, 2],
            vec![Some(5), None, Some(7), Some(8)],
            vec![
                vec![interval(0, 50)],
                vec![interval(0, 60), interval(70, 80)],
            ],
        )
    }

    fn model_with_openings_of_one_berth(openings: Vec<ClosedOpenInterval<i64>>) -> Model<i64> {
        Model::new(vec![0], vec![100], vec![1], vec![Some(5)], vec![openings])
    }

    #[test]
    fn test_counts_are_those_of_the_arrival_times_and_the_openings() {
        let model = two_by_two();
        assert_eq!(model.num_vessels(), 2);
        assert_eq!(model.num_berths(), 2);
        assert_eq!(model.vessel_iter().len(), 2);
        assert_eq!(model.berth_iter().len(), 2);
    }

    #[test]
    fn test_a_model_may_have_no_vessels_and_no_berths() {
        let model = Model::<i64>::new(vec![], vec![], vec![], vec![], vec![]);
        assert_eq!((model.num_vessels(), model.num_berths()), (0, 0));
        assert_eq!(model.vessel_iter().next(), None);
    }

    #[test]
    fn test_vessel_columns_answer_their_entries() {
        let model = two_by_two();
        let second = VesselIndex::new(1);
        assert_eq!(model.vessel_arrival_time(second), 10);
        assert_eq!(model.vessel_latest_departure_time(second), 110);
        assert_eq!(model.vessel_weight(second), 2);
    }

    #[test]
    fn test_processing_times_are_laid_out_vessel_by_vessel() {
        let model = two_by_two();
        let at = |vessel, berth| {
            model.vessel_processing_time(VesselIndex::new(vessel), BerthIndex::new(berth))
        };
        assert_eq!(
            [at(0, 0), at(0, 1), at(1, 0), at(1, 1)],
            [Some(5), None, Some(7), Some(8)]
        );
    }

    #[test]
    fn test_a_processing_time_of_zero_is_present() {
        let model = Model::new(vec![0], vec![1], vec![1], vec![Some(0)], vec![vec![]]);
        assert_eq!(
            model.vessel_processing_time(VesselIndex::new(0), BerthIndex::new(0)),
            Some(0)
        );
    }

    #[test]
    fn test_openings_are_answered_per_berth() {
        let model = two_by_two();
        assert_eq!(
            model.berth_opening_times(BerthIndex::new(0)),
            [interval(0, 50)]
        );
        assert_eq!(
            model.berth_opening_times(BerthIndex::new(1)),
            [interval(0, 60), interval(70, 80)]
        );
    }

    // Vessel 0 at berth 2 of a two-berth model lands, in the flat matrix, on vessel 1 at berth 0; the accessor must
    // refuse it in every build rather than answer that entry.
    #[test]
    #[should_panic(
        expected = "called `Model::vessel_processing_time` with Berth 2, beyond the 2 berths"
    )]
    fn test_vessel_processing_time_refuses_a_berth_beyond_the_last() {
        let _ = two_by_two().vessel_processing_time(VesselIndex::new(0), BerthIndex::new(2));
    }

    #[test]
    #[should_panic(
        expected = "called `Model::vessel_processing_time` with Vessel 2, beyond the 2 vessels"
    )]
    fn test_vessel_processing_time_refuses_a_vessel_beyond_the_last() {
        let _ = two_by_two().vessel_processing_time(VesselIndex::new(2), BerthIndex::new(0));
    }

    #[test]
    #[should_panic(
        expected = "called `Model::vessel_arrival_time` with Vessel 2, beyond the 2 vessels"
    )]
    fn test_vessel_arrival_time_refuses_a_vessel_beyond_the_last() {
        let _ = two_by_two().vessel_arrival_time(VesselIndex::new(2));
    }

    #[test]
    #[should_panic(
        expected = "called `Model::vessel_latest_departure_time` with Vessel 2, beyond the 2 vessels"
    )]
    fn test_vessel_latest_departure_time_refuses_a_vessel_beyond_the_last() {
        let _ = two_by_two().vessel_latest_departure_time(VesselIndex::new(2));
    }

    #[test]
    #[should_panic(expected = "called `Model::vessel_weight` with Vessel 2, beyond the 2 vessels")]
    fn test_vessel_weight_refuses_a_vessel_beyond_the_last() {
        let _ = two_by_two().vessel_weight(VesselIndex::new(2));
    }

    #[test]
    #[should_panic(
        expected = "called `Model::berth_opening_times` with Berth 2, beyond the 2 berths"
    )]
    fn test_berth_opening_times_refuses_a_berth_beyond_the_last() {
        let _ = two_by_two().berth_opening_times(BerthIndex::new(2));
    }

    #[test]
    fn test_try_new_refuses_latest_departure_times_not_one_per_vessel() {
        let error = Model::try_new(
            vec![0, 10],
            vec![100],
            vec![1, 2],
            vec![None; 2],
            vec![vec![]],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ModelError::LatestDepartureTimesLength {
                len: 1,
                num_vessels: 2
            }
        );
    }

    #[test]
    fn test_try_new_refuses_weights_not_one_per_vessel() {
        let error = Model::try_new(
            vec![0, 10],
            vec![100, 110],
            vec![1, 2, 3],
            vec![None; 2],
            vec![vec![]],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ModelError::WeightsLength {
                len: 3,
                num_vessels: 2
            }
        );
    }

    #[test]
    fn test_try_new_refuses_processing_times_not_one_per_pair() {
        let error = Model::try_new(
            vec![0, 10],
            vec![100, 110],
            vec![1, 2],
            vec![None; 3],
            vec![vec![], vec![]],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ModelError::ProcessingTimesLength {
                len: 3,
                num_vessels: 2,
                num_berths: 2
            }
        );
    }

    #[test]
    fn test_try_new_refuses_a_negative_processing_time_and_names_its_pair() {
        let error = Model::try_new(
            vec![0, 10],
            vec![100, 110],
            vec![1, 2],
            vec![Some(5), None, Some(-3), Some(8)],
            vec![vec![], vec![]],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ModelError::NegativeProcessingTime {
                vessel: VesselIndex::new(1),
                berth: BerthIndex::new(0),
                processing_time: -3,
            }
        );
    }

    #[test]
    #[should_panic(expected = "called `Model::new` with 1 weights for 2 vessels")]
    fn test_new_panics_with_the_refusal_of_try_new() {
        let _ = Model::new(
            vec![0, 10],
            vec![100, 110],
            vec![1],
            vec![None; 2],
            vec![vec![]],
        );
    }

    #[test]
    fn test_ids_can_name_counts_up_to_u32_max() {
        let largest = usize::try_from(u32::MAX).expect("a usize should hold every u32");
        assert!(ids_can_name(0));
        assert!(ids_can_name(largest));
        #[cfg(target_pointer_width = "64")]
        assert!(!ids_can_name(largest + 1));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_check_counts_refuses_more_vessels_or_berths_than_ids_can_name() {
        let beyond = 1 << 32;
        assert_eq!(
            check_counts::<i64>(beyond, 1),
            Err(ModelError::TooManyVessels { count: beyond })
        );
        assert_eq!(
            check_counts::<i64>(1, beyond),
            Err(ModelError::TooManyBerths { count: beyond })
        );
        assert_eq!(check_counts::<i64>(1, 1), Ok(()));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_processing_time_column_refuses_a_product_beyond_usize() {
        let error = processing_time_column::<i64>(&[], usize::MAX, 2).unwrap_err();
        assert_eq!(
            error,
            ModelError::ProcessingTimesLength {
                len: 0,
                num_vessels: usize::MAX,
                num_berths: 2
            }
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_every_error_states_what_went_wrong_with_its_values() {
        let cases: [(ModelError<i64>, &str); 6] = [
            (
                ModelError::TooManyVessels { count: 1 << 33 },
                "8589934592 vessels, more than the 4294967295 that ids can name",
            ),
            (
                ModelError::TooManyBerths { count: 1 << 33 },
                "8589934592 berths, more than the 4294967295 that ids can name",
            ),
            (
                ModelError::LatestDepartureTimesLength {
                    len: 1,
                    num_vessels: 2,
                },
                "1 latest departure times for 2 vessels",
            ),
            (
                ModelError::WeightsLength {
                    len: 3,
                    num_vessels: 2,
                },
                "3 weights for 2 vessels",
            ),
            (
                ModelError::ProcessingTimesLength {
                    len: 3,
                    num_vessels: 2,
                    num_berths: 2,
                },
                "3 processing times for 2 vessels and 2 berths, which need one per pair",
            ),
            (
                ModelError::NegativeProcessingTime {
                    vessel: VesselIndex::new(1),
                    berth: BerthIndex::new(0),
                    processing_time: -3,
                },
                "negative processing time -3 of Vessel 1 at Berth 0",
            ),
        ];
        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
        }
    }

    #[test]
    fn test_processing_time_stores_absence_apart_from_every_present_time() {
        assert_eq!(ProcessingTime::new(Some(0i64)).get(), Some(0));
        assert_eq!(ProcessingTime::new(Some(i64::MAX)).get(), Some(i64::MAX));
        assert_eq!(ProcessingTime::<i64>::new(None).get(), None);
        assert_eq!(format!("{:?}", ProcessingTime::<i64>::new(None)), "None");
    }

    #[test]
    fn test_openings_out_of_order_are_kept_in_order() {
        let model = model_with_openings_of_one_berth(vec![interval(20, 30), interval(0, 10)]);
        assert_eq!(
            model.berth_opening_times(BerthIndex::new(0)),
            [interval(0, 10), interval(20, 30)]
        );
    }

    // A berth open on [0, 10) and on [5, 20) is open on [0, 20): a vessel of length 15 fits at 0.
    #[test]
    fn test_overlapping_openings_are_kept_as_their_union() {
        let model = model_with_openings_of_one_berth(vec![interval(5, 20), interval(0, 10)]);
        assert_eq!(
            model.berth_opening_times(BerthIndex::new(0)),
            [interval(0, 20)]
        );
    }

    // A berth open on [0, 10) and on [10, 20) is open on [0, 20) without a break.
    #[test]
    fn test_touching_openings_are_kept_as_their_union() {
        let model = model_with_openings_of_one_berth(vec![interval(0, 10), interval(10, 20)]);
        assert_eq!(
            model.berth_opening_times(BerthIndex::new(0)),
            [interval(0, 20)]
        );
    }

    #[test]
    fn test_empty_openings_are_dropped() {
        let model = model_with_openings_of_one_berth(vec![interval(5, 5), interval(10, 20)]);
        assert_eq!(
            model.berth_opening_times(BerthIndex::new(0)),
            [interval(10, 20)]
        );
    }

    #[test]
    fn test_openings_of_berths_without_any_are_empty() {
        let model = Model::<i64>::new(
            vec![],
            vec![],
            vec![],
            vec![],
            vec![vec![], vec![interval(0, 5)], vec![]],
        );
        assert_eq!(model.num_berths(), 3);
        assert_eq!(model.berth_opening_times(BerthIndex::new(0)), []);
        assert_eq!(
            model.berth_opening_times(BerthIndex::new(1)),
            [interval(0, 5)]
        );
        assert_eq!(model.berth_opening_times(BerthIndex::new(2)), []);
    }

    /// The integers in `[0, 40)` that one of `intervals` holds: the model the union is checked against.
    fn covered_points(intervals: &[ClosedOpenInterval<i64>]) -> Vec<i64> {
        (0..40)
            .filter(|&point| {
                intervals
                    .iter()
                    .any(|interval| interval.contains_point(point))
            })
            .collect()
    }

    fn intervals_within_forty() -> impl Strategy<Value = Vec<ClosedOpenInterval<i64>>> {
        prop::collection::vec((0i64..40, 0i64..40), 0..8).prop_map(|bounds| {
            bounds
                .into_iter()
                .map(|(a, b)| ClosedOpenInterval::new(a.min(b), a.max(b)))
                .collect()
        })
    }

    /// A small model, with a processing time absent wherever its seed is odd.
    fn small_model() -> impl Strategy<Value = Model<i64>> {
        (0usize..5, 0usize..5).prop_flat_map(|(num_vessels, num_berths)| {
            (
                prop::collection::vec(0i64..100, num_vessels),
                prop::collection::vec(0i64..100, num_vessels),
                prop::collection::vec(0i64..100, num_vessels),
                prop::collection::vec(0i64..100, num_vessels * num_berths),
                prop::collection::vec(intervals_within_forty(), num_berths),
            )
                .prop_map(|(arrivals, departures, weights, seeds, openings)| {
                    let processing = seeds
                        .into_iter()
                        .map(|seed| (seed % 2 == 0).then_some(seed))
                        .collect();
                    Model::new(arrivals, departures, weights, processing, openings)
                })
        })
    }

    fn refuses<R>(access: impl FnOnce() -> R) -> bool {
        catch_unwind(AssertUnwindSafe(access)).is_err()
    }

    proptest! {
        #[test]
        fn test_union_of_openings_covers_the_same_points_in_the_fewest_ordered_intervals(
            openings in intervals_within_forty(),
        ) {
            let union = union_of_openings(&openings);
            prop_assert_eq!(covered_points(&union), covered_points(&openings));
            prop_assert!(union.iter().all(|interval| !interval.is_empty()));
            prop_assert!(union.windows(2).all(|pair| pair[0].end() < pair[1].start()));
        }

        #[test]
        fn test_checked_and_unchecked_accessors_agree_on_every_id_of_the_model(model in small_model()) {
            for vessel in model.vessel_iter() {
                // SAFETY: `vessel` is one of those the model's own walk yields.
                let unchecked = unsafe {
                    (
                        model.vessel_arrival_time_unchecked(vessel),
                        model.vessel_latest_departure_time_unchecked(vessel),
                        model.vessel_weight_unchecked(vessel),
                    )
                };
                let checked = (
                    model.vessel_arrival_time(vessel),
                    model.vessel_latest_departure_time(vessel),
                    model.vessel_weight(vessel),
                );
                prop_assert_eq!(checked, unchecked);
                for berth in model.berth_iter() {
                    // SAFETY: `vessel` and `berth` are among those the model's own walks yield.
                    let unchecked = unsafe { model.vessel_processing_time_unchecked(vessel, berth) };
                    prop_assert_eq!(model.vessel_processing_time(vessel, berth), unchecked);
                }
            }
            for berth in model.berth_iter() {
                // SAFETY: `berth` is one of those the model's own walk yields.
                let unchecked = unsafe { model.berth_opening_times_unchecked(berth) };
                prop_assert_eq!(model.berth_opening_times(berth), unchecked);
            }
        }

        #[test]
        fn test_checked_accessors_refuse_every_id_beyond_the_model(model in small_model(), beyond in 0usize..3) {
            let vessel = VesselIndex::new(model.num_vessels() + beyond);
            let berth = BerthIndex::new(model.num_berths() + beyond);
            prop_assert!(refuses(|| model.vessel_arrival_time(vessel)));
            prop_assert!(refuses(|| model.vessel_latest_departure_time(vessel)));
            prop_assert!(refuses(|| model.vessel_weight(vessel)));
            prop_assert!(refuses(|| model.vessel_processing_time(vessel, BerthIndex::new(0))));
            prop_assert!(refuses(|| model.vessel_processing_time(VesselIndex::new(0), berth)));
            prop_assert!(refuses(|| model.berth_opening_times(berth).len()));
        }
    }
}
