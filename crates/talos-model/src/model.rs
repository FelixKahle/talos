// Copyright (c) 2026 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use crate::index::{BerthIndex, VesselIndex};
use talos_core::{math::interval::ClosedOpenInterval, utils::num::SolverNumeric};

/// A processing time that may be absent.
///
/// Instead of using `Option<T>`, this type uses a sentinel encoding to avoid
/// the additional discriminant that `Option` typically introduces for integer
/// types. In hot loops and dense collections, keeping the value to a single
/// machine word can improve cache locality and reduce memory traffic.
///
/// Encoding:
/// - Non-negative values (>= 0) represent a concrete processing time.
/// - Negative values (<= -1) are reserved to indicate absence.
///
/// This convention assumes valid processing times are non-negative. If negative
/// values are meaningful in your domain, use `Option<T>` instead.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessingTime<T>(T);

impl<T> ProcessingTime<T>
where
    T: SolverNumeric,
{
    const NONE_SENTINEL: T = T::NEGATIVE_ONE;

    /// Creates a `ProcessingTime` from an `Option<T>`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(5i32));
    /// assert!(some_time.is_some());
    /// assert_eq!(some_time.raw(), 5);
    /// ```
    #[inline]
    pub fn from_option(value: Option<T>) -> Self {
        match value {
            Some(v) => ProcessingTime(v),
            None => ProcessingTime(Self::NONE_SENTINEL),
        }
    }

    /// Creates a `ProcessingTime` from a raw value without checking for sentinel.
    /// If you pass a negative value, it will be treated as `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let time = ProcessingTime::from_raw(10i32);
    /// assert!(time.is_some());
    /// assert_eq!(time.raw(), 10);
    /// ```
    #[inline]
    pub const fn from_raw(value: T) -> Self {
        ProcessingTime(value)
    }

    /// Creates a `ProcessingTime` representing `Some`.
    ///
    /// # Panics
    ///
    /// This function will panic if the provided value is negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::some(5i32);
    /// assert!(some_time.is_some());
    /// assert_eq!(some_time.raw(), 5);
    /// ```
    pub fn some(value: T) -> Self
    where
        T: PartialOrd + std::fmt::Display,
    {
        assert!(
            value > Self::NONE_SENTINEL,
            "called `ProcessingTime::some` with a negative value: {}",
            value
        );

        ProcessingTime(value)
    }

    /// Creates a `ProcessingTime` representing `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert!(none_time.is_none());
    /// ```
    #[inline]
    pub fn none() -> Self {
        ProcessingTime(Self::NONE_SENTINEL)
    }

    /// Checks if the `ProcessingTime` represents `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert!(none_time.is_none());
    /// ```
    #[inline]
    pub fn is_none(&self) -> bool
    where
        T: PartialOrd,
    {
        self.0 <= Self::NONE_SENTINEL
    }

    /// Checks if the `ProcessingTime` represents `Some`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(3i32));
    /// assert!(some_time.is_some());
    /// ```
    #[inline]
    pub fn is_some(&self) -> bool
    where
        T: PartialOrd,
    {
        !self.is_none()
    }

    /// Returns the raw value, including sentinel if present.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let time = ProcessingTime::from_option(Some(7i32));
    /// assert_eq!(time.raw(), 7);
    /// ```
    #[inline]
    pub fn raw(&self) -> T {
        self.0
    }

    /// Converts the `ProcessingTime` back into an `Option<T>`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(4i32));
    /// assert_eq!(some_time.into_option(), Some(4));
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert_eq!(none_time.into_option(), None);
    /// ```
    #[inline]
    pub fn into_option(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        if self.is_none() { None } else { Some(self.0) }
    }

    /// Unwraps the `ProcessingTime`, panicking if it is `None`.
    ///
    /// # Panics
    ///
    /// This function will panic if called on a `ProcessingTime` that represents `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(6i32));
    /// assert_eq!(some_time.unwrap(), 6);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// // The following line would panic:
    /// // none_time.unwrap();
    /// ```
    pub fn unwrap(&self) -> T
    where
        T: PartialOrd,
    {
        if self.is_none() {
            panic!("called `ProcessingTime::unwrap()` on a `None` value")
        }
        self.0
    }

    /// Unwraps the `ProcessingTime` without checking for `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(6i32));
    /// assert_eq!(some_time.unwrap_unchecked(), 6);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// // The following line will NOT panic, but yields an invalid value:
    /// // none_time.unwrap_unchecked();
    /// ```
    pub fn unwrap_unchecked(&self) -> T {
        self.0
    }

    /// Unwraps the `ProcessingTime`, returning a default value if it is `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(8i32));
    /// assert_eq!(some_time.unwrap_or(0), 8);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert_eq!(none_time.unwrap_or(0), 0);
    /// ```
    #[inline]
    pub fn unwrap_or(&self, default: T) -> T
    where
        T: PartialOrd,
    {
        if self.is_none() { default } else { self.0 }
    }

    /// Unwraps the `ProcessingTime`, computing a default value if it is `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use talos_model::model::ProcessingTime;
    ///
    /// let some_time = ProcessingTime::from_option(Some(9i32));
    /// assert_eq!(some_time.unwrap_or_else(|| 1 + 1), 9);
    ///
    /// let none_time: ProcessingTime<i32> = ProcessingTime::none();
    /// assert_eq!(none_time.unwrap_or_else(|| 1 + 1), 2);
    /// ```
    #[inline]
    pub fn unwrap_or_else<F>(&self, f: F) -> T
    where
        T: PartialOrd,
        F: FnOnce() -> T,
    {
        if self.is_none() { f() } else { self.0 }
    }
}

impl<T> std::fmt::Debug for ProcessingTime<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "ProcessingTime(None)")
        } else {
            write!(f, "ProcessingTime(Some({:?}))", self.0)
        }
    }
}

impl<T> std::fmt::Display for ProcessingTime<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "ProcessingTime(None)")
        } else {
            write!(f, "ProcessingTime({})", self.0)
        }
    }
}

impl<T> From<Option<T>> for ProcessingTime<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn from(value: Option<T>) -> Self {
        ProcessingTime::from_option(value)
    }
}

impl<T> From<ProcessingTime<T>> for Option<T>
where
    T: SolverNumeric,
{
    #[inline]
    fn from(val: ProcessingTime<T>) -> Self {
        val.into_option()
    }
}

#[inline(always)]
fn flatten_index(num_berths: usize, vessel_index: VesselIndex, berth_index: BerthIndex) -> usize {
    vessel_index.get() * num_berths + berth_index.get()
}

pub struct Model<T>
where
    T: SolverNumeric,
{
    arrival_times: Vec<T>,                    // len = num_vessels
    latest_departure_times: Vec<T>,           // len = num_vessels
    vessel_weights: Vec<T>,                   // len = num_vessels
    processing_times: Vec<ProcessingTime<T>>, // len = num_vessels * num_berths
    opening_intervals: Vec<ClosedOpenInterval<T>>,
    opening_offsets: Vec<usize>, // len = num_berths + 1
}

impl<T> Model<T>
where
    T: SolverNumeric,
{
    /// Creates a new `Model` with the specified parameters.
    /// All vectors must have lengths consistent with `num_vessels` and `num_berths`.
    /// - `arrival_times`, `latest_departure_times`, and `vessel_weights` must have length `num_vessels`.
    /// - `processing_times` must have length `num_vessels * num_berths`.
    /// - `opening_times` must have length `num_berths`.
    ///
    /// # Panics
    ///
    /// Panics if any of the input vectors do not have the expected length based on `num_vessels` and `num_berths`.
    #[inline]
    pub fn new(
        num_vessels: usize,
        num_berths: usize,
        arrival_times: Vec<T>,
        latest_departure_times: Vec<T>,
        vessel_weights: Vec<T>,
        processing_times: Vec<ProcessingTime<T>>,
        opening_times: Vec<Vec<ClosedOpenInterval<T>>>,
    ) -> Self {
        assert_eq!(
            arrival_times.len(),
            num_vessels,
            "called `Model::new` with `arrival_times` length {} but expected {}",
            arrival_times.len(),
            num_vessels
        );
        assert_eq!(
            latest_departure_times.len(),
            num_vessels,
            "called `Model::new` with `latest_departure_times` length {} but expected {}",
            latest_departure_times.len(),
            num_vessels
        );
        assert_eq!(
            vessel_weights.len(),
            num_vessels,
            "called `Model::new` with `vessel_weights` length {} but expected {}",
            vessel_weights.len(),
            num_vessels
        );
        assert_eq!(
            processing_times.len(),
            num_vessels * num_berths,
            "called `Model::new` with `processing_times` length {} but expected {}",
            processing_times.len(),
            num_vessels * num_berths
        );
        assert_eq!(
            opening_times.len(),
            num_berths,
            "called `Model::new` with `opening_times` length {} but expected {}",
            opening_times.len(),
            num_berths
        );

        let mut opening_intervals = Vec::new();
        let mut opening_offsets = Vec::with_capacity(num_berths + 1);

        let mut current_offset = 0;
        opening_offsets.push(current_offset);

        for berth_intervals in opening_times {
            opening_intervals.extend_from_slice(&berth_intervals);
            current_offset += berth_intervals.len();
            opening_offsets.push(current_offset);
        }

        Model {
            arrival_times,
            latest_departure_times,
            vessel_weights,
            processing_times,
            opening_intervals,
            opening_offsets,
        }
    }

    /// Overrides the current model state with new data.
    /// This method reuses the internal memory capacities of the vectors to avoid
    /// heap allocations whenever possible.
    ///
    /// # Panics
    ///
    /// Panics if any of the input slices do not have the expected length based on
    /// `num_vessels` and `num_berths`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn override_from(
        &mut self,
        num_vessels: usize,
        num_berths: usize,
        arrival_times: &[T],
        latest_departure_times: &[T],
        vessel_weights: &[T],
        processing_times: &[ProcessingTime<T>],
        opening_times: &[&[ClosedOpenInterval<T>]],
    ) {
        assert_eq!(
            arrival_times.len(),
            num_vessels,
            "called `Model::override_from` with `arrival_times` length {} but expected {}",
            arrival_times.len(),
            num_vessels
        );
        assert_eq!(
            latest_departure_times.len(),
            num_vessels,
            "called `Model::override_from` with `latest_departure_times` length {} but expected {}",
            latest_departure_times.len(),
            num_vessels
        );
        assert_eq!(
            vessel_weights.len(),
            num_vessels,
            "called `Model::override_from` with `vessel_weights` length {} but expected {}",
            vessel_weights.len(),
            num_vessels
        );
        assert_eq!(
            processing_times.len(),
            num_vessels * num_berths,
            "called `Model::override_from` with `processing_times` length {} but expected {}",
            processing_times.len(),
            num_vessels * num_berths
        );
        assert_eq!(
            opening_times.len(),
            num_berths,
            "called `Model::override_from` with `opening_times` length {} but expected {}",
            opening_times.len(),
            num_berths
        );

        self.arrival_times.clear();
        self.arrival_times.extend_from_slice(arrival_times);

        self.latest_departure_times.clear();
        self.latest_departure_times
            .extend_from_slice(latest_departure_times);

        self.vessel_weights.clear();
        self.vessel_weights.extend_from_slice(vessel_weights);

        self.processing_times.clear();
        self.processing_times.extend_from_slice(processing_times);

        // Rebuild the flattened intervals and offsets, reusing existing capacity
        self.opening_intervals.clear();
        self.opening_offsets.clear();

        let mut current_offset = 0;
        self.opening_offsets.push(current_offset);

        for berth_intervals in opening_times {
            self.opening_intervals.extend_from_slice(berth_intervals);
            current_offset += berth_intervals.len();
            self.opening_offsets.push(current_offset);
        }
    }

    /// Returns the number of vessels in the model.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.arrival_times.len()
    }

    /// Returns the number of berths in the model.
    ///
    /// # Examples
    #[inline]
    pub fn num_berths(&self) -> usize {
        // The offsets array is exactly num_berths + 1 in length.
        self.opening_offsets.len() - 1
    }

    /// Returns a slice of all arrival times.
    #[inline]
    pub fn vessel_arrival_times(&self) -> &[T] {
        &self.arrival_times
    }

    /// Returns a slice of all vessel weights.
    #[inline]
    pub fn vessel_weights(&self) -> &[T] {
        &self.vessel_weights
    }

    /// Returns a slice of all latest departure times.
    #[inline]
    pub fn vessel_latest_departure_times(&self) -> &[T] {
        &self.latest_departure_times
    }

    /// Returns a slice of all processing times.
    #[inline]
    pub fn vessel_processing_times_matrix(&self) -> &[ProcessingTime<T>] {
        &self.processing_times
    }

    /// Returns a slice of processing times for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    #[inline]
    pub fn vessel_processing_times(&self, vessel_index: VesselIndex) -> &[ProcessingTime<T>] {
        let start = vessel_index.get() * self.num_berths();
        let end = start + self.num_berths();
        &self.processing_times[start..end]
    }

    /// Returns the arrival time for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    #[inline]
    pub fn vessel_arrival_time(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(index < self.num_vessels());

        self.arrival_times[index]
    }

    /// Returns the arrival time for the specified vessel without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    #[inline]
    pub unsafe fn vessel_arrival_time_unchecked(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(index < self.num_vessels());

        unsafe { *self.arrival_times.get_unchecked(index) }
    }

    /// Returns the latest departure time for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    #[inline]
    pub fn vessel_latest_departure_time(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(index < self.num_vessels());

        self.latest_departure_times[index]
    }

    /// Returns the latest departure time for the specified vessel without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    #[inline]
    pub unsafe fn vessel_latest_departure_time_unchecked(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(index < self.num_vessels());

        unsafe { *self.latest_departure_times.get_unchecked(index) }
    }

    /// Returns the weight for the specified vessel.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    #[inline]
    pub fn vessel_weight(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(index < self.num_vessels());

        self.vessel_weights[index]
    }

    /// Returns the weight for the specified vessel without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    #[inline]
    pub unsafe fn vessel_weight_unchecked(&self, vessel_index: VesselIndex) -> T {
        let index = vessel_index.get();
        debug_assert!(index < self.num_vessels());

        unsafe { *self.vessel_weights.get_unchecked(index) }
    }

    /// Returns the processing time for the specified (vessel, berth) pair.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()` or
    /// if `berth_index` is not in `0..num_berths()`.
    #[inline]
    pub fn vessel_processing_time(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> ProcessingTime<T> {
        debug_assert!(vessel_index < self.num_vessels());
        debug_assert!(berth_index < self.num_berths());

        let flat_index = flatten_index(self.num_berths(), vessel_index, berth_index);
        debug_assert!(flat_index < self.processing_times.len());

        self.processing_times[flat_index]
    }

    /// Returns the processing time for the specified (vessel, berth) pair without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index` and `berth_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()` and
    /// `berth_index` is in `0..num_berths()`. Undefined behavior
    /// may occur if this precondition is violated.
    #[inline]
    pub unsafe fn vessel_processing_time_unchecked(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> ProcessingTime<T> {
        debug_assert!(vessel_index < self.num_vessels());
        debug_assert!(berth_index < self.num_berths());

        let flat_index = flatten_index(self.num_berths(), vessel_index, berth_index);
        debug_assert!(flat_index < self.processing_times.len());

        unsafe { *self.processing_times.get_unchecked(flat_index) }
    }

    /// Returns `true` if the specified vessel is allowed to dock at the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is not in `0..num_vessels()`.
    #[inline]
    pub fn vessel_allowed_on_berth(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> bool {
        let index = vessel_index.get();
        debug_assert!(index < self.num_vessels());

        self.vessel_processing_time(vessel_index, berth_index)
            .is_some()
    }

    /// Returns `true` if the specified vessel is allowed to dock at the specified berth without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `vessel_index`.
    /// The caller must ensure that `vessel_index` is in `0..num_vessels()`. Undefined behavior
    /// may occur if this precondition is violated.
    #[inline]
    pub unsafe fn vessel_allowed_on_berth_unchecked(
        &self,
        vessel_index: VesselIndex,
        berth_index: BerthIndex,
    ) -> bool {
        let index = vessel_index.get();
        debug_assert!(index < self.num_vessels());

        unsafe { self.vessel_processing_time_unchecked(vessel_index, berth_index) }.is_some()
    }

    /// Returns the opening times for the specified berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is not in `0..num_berths()`.
    #[inline]
    pub fn berth_opening_times(&self, berth_index: BerthIndex) -> &[ClosedOpenInterval<T>] {
        let index = berth_index.get();
        debug_assert!(index < self.num_berths());

        let start = self.opening_offsets[index];
        let end = self.opening_offsets[index + 1];
        &self.opening_intervals[start..end]
    }

    /// Returns the opening times for the specified berth without bounds checking.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not perform bounds checking on `berth_index`.
    /// The caller must ensure that `berth_index` is in `0..num_berths()`. Undefined behavior
    /// may occur if this precondition is violated.
    #[inline]
    pub unsafe fn berth_opening_times_unchecked(
        &self,
        berth_index: BerthIndex,
    ) -> &[ClosedOpenInterval<T>] {
        let index = berth_index.get();
        debug_assert!(index < self.num_berths());

        let start = *unsafe { self.opening_offsets.get_unchecked(index) };
        let end = *unsafe { self.opening_offsets.get_unchecked(index + 1) };
        unsafe { self.opening_intervals.get_unchecked(start..end) }
    }
}

impl<T> std::fmt::Debug for Model<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("arrival_times", &self.arrival_times)
            .field("latest_departure_times", &self.latest_departure_times)
            .field("vessel_weights", &self.vessel_weights)
            .finish()
    }
}

impl<T> std::fmt::Display for Model<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Model(num_vessels: {}, num_berths: {})",
            self.num_vessels(),
            self.num_berths()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vessel_index(index: usize) -> VesselIndex {
        VesselIndex::new(index)
    }

    fn berth_index(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    #[test]
    fn test_processing_time_creation() {
        // Test Some
        let pt_some = ProcessingTime::some(10i64);
        assert!(pt_some.is_some());
        assert!(!pt_some.is_none());
        assert_eq!(pt_some.raw(), 10);
        assert_eq!(pt_some.into_option(), Some(10));

        // Test None
        let pt_none = ProcessingTime::<i64>::none();
        assert!(pt_none.is_none());
        assert!(!pt_none.is_some());
        assert_eq!(pt_none.raw(), -1);
        assert_eq!(pt_none.into_option(), None);
    }

    #[test]
    #[should_panic(expected = "called `ProcessingTime::some` with a negative value")]
    fn test_processing_time_some_panic_on_negative() {
        ProcessingTime::some(-5i64);
    }

    #[test]
    fn test_processing_time_conversions() {
        let pt1 = ProcessingTime::from_option(Some(42i64));
        assert_eq!(pt1.unwrap(), 42);

        let pt2 = ProcessingTime::from_option(None::<i64>);
        assert!(pt2.is_none());

        let pt3 = ProcessingTime::from_raw(5i64);
        assert_eq!(pt3.into_option(), Some(5));

        let pt4 = ProcessingTime::from_raw(-1i64);
        assert!(pt4.is_none());
    }

    #[test]
    fn test_processing_time_unwraps() {
        let pt_some = ProcessingTime::some(20i64);
        let pt_none = ProcessingTime::<i64>::none();

        assert_eq!(pt_some.unwrap_or(0), 20);
        assert_eq!(pt_none.unwrap_or(0), 0);

        assert_eq!(pt_some.unwrap_or_else(|| 100), 20);
        assert_eq!(pt_none.unwrap_or_else(|| 100), 100);
    }

    #[test]
    #[should_panic(expected = "called `ProcessingTime::unwrap()` on a `None` value")]
    fn test_processing_time_unwrap_panic() {
        let pt_none = ProcessingTime::<i64>::none();
        pt_none.unwrap();
    }

    /// Helper to build a valid 2-vessel, 2-berth model.
    fn build_valid_model() -> Model<i64> {
        let num_vessels = 2;
        let num_berths = 2;

        let arrivals = vec![0, 10];
        let departures = vec![100, 110];
        let weights = vec![1, 2];

        // Matrix: [v0b0, v0b1, v1b0, v1b1]
        let processing = vec![
            ProcessingTime::some(5),
            ProcessingTime::none(), // Vessel 0
            ProcessingTime::some(7),
            ProcessingTime::some(8), // Vessel 1
        ];

        let opening = vec![
            vec![ClosedOpenInterval::new(0, 50)], // Berth 0
            vec![ClosedOpenInterval::new(0, 60)], // Berth 1
        ];

        Model::new(
            num_vessels,
            num_berths,
            arrivals,
            departures,
            weights,
            processing,
            opening,
        )
    }

    #[test]
    fn test_model_construction_valid() {
        let model = build_valid_model();
        assert_eq!(model.num_vessels(), 2);
        assert_eq!(model.num_berths(), 2);
    }

    #[test]
    #[should_panic]
    fn test_model_construction_invalid_departures_length() {
        Model::new(
            2,
            2,
            vec![0, 10],
            vec![100], // INVALID
            vec![1, 2],
            vec![ProcessingTime::none(); 4],
            vec![vec![]; 2],
        );
    }

    #[test]
    #[should_panic]
    fn test_model_construction_invalid_processing_length() {
        Model::new(
            2,
            2,
            vec![0, 10],
            vec![100, 110],
            vec![1, 2],
            vec![ProcessingTime::none(); 3], // INVALID: expected 2 * 2 = 4
            vec![vec![]; 2],
        );
    }

    #[test]
    fn test_model_getters_1d() {
        let model = build_valid_model();

        assert_eq!(model.vessel_arrival_time(vessel_index(0)), 0);
        assert_eq!(model.vessel_arrival_time(vessel_index(1)), 10);

        assert_eq!(model.vessel_latest_departure_time(vessel_index(0)), 100);
        assert_eq!(model.vessel_weight(vessel_index(1)), 2);

        // Unchecked access (Safe in context of tests)
        unsafe {
            assert_eq!(model.vessel_arrival_time_unchecked(vessel_index(0)), 0);
        }
    }

    #[test]
    fn test_model_processing_times_matrix_flattening() {
        let model = build_valid_model();

        // Vessel 0, Berth 0 -> 5
        assert_eq!(
            model
                .vessel_processing_time(vessel_index(0), berth_index(0))
                .unwrap(),
            5
        );
        // Vessel 0, Berth 1 -> None
        assert!(
            model
                .vessel_processing_time(vessel_index(0), berth_index(1))
                .is_none()
        );
        // Vessel 1, Berth 0 -> 7
        assert_eq!(
            model
                .vessel_processing_time(vessel_index(1), berth_index(0))
                .unwrap(),
            7
        );
        // Vessel 1, Berth 1 -> 8
        assert_eq!(
            model
                .vessel_processing_time(vessel_index(1), berth_index(1))
                .unwrap(),
            8
        );
    }

    #[test]
    fn test_model_vessel_processing_times_slice() {
        let model = build_valid_model();

        // Vessel 0 should get slice [Some(5), None]
        let v0_slice = model.vessel_processing_times(vessel_index(0));
        assert_eq!(v0_slice.len(), 2);
        assert_eq!(v0_slice[0].into_option(), Some(5));
        assert!(v0_slice[1].is_none());

        // Vessel 1 should get slice [Some(7), Some(8)]
        let v1_slice = model.vessel_processing_times(vessel_index(1));
        assert_eq!(v1_slice.len(), 2);
        assert_eq!(v1_slice[0].into_option(), Some(7));
        assert_eq!(v1_slice[1].into_option(), Some(8));
    }

    #[test]
    fn test_model_allowed_on_berth() {
        let model = build_valid_model();

        assert!(model.vessel_allowed_on_berth(vessel_index(0), berth_index(0))); // Some(5)
        assert!(!model.vessel_allowed_on_berth(vessel_index(0), berth_index(1))); // None

        unsafe {
            assert!(model.vessel_allowed_on_berth_unchecked(vessel_index(1), berth_index(1)));
            // Some(8)
        }
    }

    #[test]
    fn test_model_berth_opening_times() {
        let model = build_valid_model();

        let b0_openings = model.berth_opening_times(berth_index(0));
        assert_eq!(b0_openings.len(), 1);
        assert_eq!(b0_openings[0].start(), 0);
        assert_eq!(b0_openings[0].end(), 50);

        let b1_openings = model.berth_opening_times(berth_index(1));
        assert_eq!(b1_openings.len(), 1);
        assert_eq!(b1_openings[0].end(), 60);
    }
}
