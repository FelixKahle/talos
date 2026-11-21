// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Closed-open intervals `[start, end)` of integers.

use num_traits::PrimInt;
use std::cmp::{max, min};
use std::fmt;
use std::iter::FusedIterator;
use std::ops::{Bound, Range, RangeBounds};

/// The integers from `start` up to but not including `end`: the interval `[start, end)`.
///
/// Availability windows, maintenance gaps and processing spans are intervals of this kind. The closed-open form
/// composes without off-by-one corrections: `[a, b)` and `[b, c)` meet without overlap, and the length is
/// `end - start`.
///
/// An interval always has `start <= end`; every way of making one establishes it. An empty interval
/// (`start == end`) keeps its position, and the predicates and operations below work on the bounds, so an empty
/// interval at 20 is adjacent to `[10, 20)` but has no union with `[0, 10)`.
///
/// # Examples
///
/// ```rust
/// use talos_core::ClosedOpenInterval;
///
/// let morning = ClosedOpenInterval::new(8i64, 12);
/// let shift = ClosedOpenInterval::new(10i64, 18);
///
/// assert_eq!(morning.intersection(shift), Some(ClosedOpenInterval::new(10, 12)));
/// assert_eq!(morning.union(shift), Some(ClosedOpenInterval::new(8, 18)));
/// assert_eq!(morning.iter().collect::<Vec<_>>(), [8, 9, 10, 11]);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ClosedOpenInterval<T>
where
    T: PrimInt,
{
    start_inclusive: T,
    end_exclusive: T,
}

impl<T> ClosedOpenInterval<T>
where
    T: PrimInt,
{
    /// The interval `[start_inclusive, end_exclusive)`.
    ///
    /// # Panics
    ///
    /// Panics if `start_inclusive` is greater than `end_exclusive`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let interval = ClosedOpenInterval::new(0, 10);
    /// assert_eq!(interval.len(), 10);
    /// ```
    #[inline]
    #[track_caller]
    pub fn new(start_inclusive: T, end_exclusive: T) -> Self
    where
        T: fmt::Debug,
    {
        match Self::try_new(start_inclusive, end_exclusive) {
            Some(interval) => interval,
            None => panic_start_after_end(start_inclusive, end_exclusive),
        }
    }

    /// The interval `[start_inclusive, end_exclusive)`, or `None` if `start_inclusive` is greater than
    /// `end_exclusive`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// assert!(ClosedOpenInterval::try_new(0, 10).is_some());
    /// assert!(ClosedOpenInterval::try_new(10, 0).is_none());
    /// ```
    #[inline]
    pub fn try_new(start_inclusive: T, end_exclusive: T) -> Option<Self> {
        (start_inclusive <= end_exclusive)
            .then(|| Self::from_ordered(start_inclusive, end_exclusive))
    }

    /// The interval `[start_inclusive, end_exclusive)` from bounds the caller has already ordered.
    #[inline]
    fn from_ordered(start_inclusive: T, end_exclusive: T) -> Self {
        debug_assert!(
            start_inclusive <= end_exclusive,
            "called `ClosedOpenInterval::from_ordered` with a start after the end"
        );
        Self {
            start_inclusive,
            end_exclusive,
        }
    }

    /// The least integer of the interval, or its position if it is empty.
    #[inline]
    pub const fn start(&self) -> T {
        self.start_inclusive
    }

    /// The first integer after the interval.
    #[inline]
    pub const fn end(&self) -> T {
        self.end_exclusive
    }

    /// Whether the two intervals share an integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// assert!(a.intersects(ClosedOpenInterval::new(5, 15)));
    /// assert!(!a.intersects(ClosedOpenInterval::new(10, 20)));
    /// ```
    #[inline]
    pub fn intersects(&self, other: Self) -> bool {
        // Both bounds must be compared together: an empty interval inside another starts before the other ends
        // and ends after it starts, yet shares no integer with it.
        max(self.start_inclusive, other.start_inclusive)
            < min(self.end_exclusive, other.end_exclusive)
    }

    /// Whether one interval ends where the other starts.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// assert!(a.adjacent(ClosedOpenInterval::new(10, 20)));
    /// assert!(!a.adjacent(ClosedOpenInterval::new(9, 20)));
    /// ```
    #[inline]
    pub fn adjacent(&self, other: Self) -> bool {
        self.end_exclusive == other.start_inclusive || other.end_exclusive == self.start_inclusive
    }

    /// Whether a gap of at least one integer separates the two intervals: they neither intersect nor touch.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// assert!(a.disjoint(ClosedOpenInterval::new(15, 20)));
    /// assert!(!a.disjoint(ClosedOpenInterval::new(10, 15)));
    /// ```
    #[inline]
    pub fn disjoint(&self, other: Self) -> bool {
        !self.intersects_or_adjacent(other)
    }

    /// Whether the two intervals intersect or touch, so that together they cover one interval.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// assert!(a.intersects_or_adjacent(ClosedOpenInterval::new(10, 20)));
    /// assert!(!a.intersects_or_adjacent(ClosedOpenInterval::new(12, 20)));
    /// ```
    #[inline]
    pub fn intersects_or_adjacent(&self, other: Self) -> bool {
        self.start_inclusive <= other.end_exclusive && other.start_inclusive <= self.end_exclusive
    }

    /// Whether `value` lies in the interval.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let interval = ClosedOpenInterval::new(0, 10);
    /// assert!(interval.contains_point(0));
    /// assert!(!interval.contains_point(10));
    /// ```
    #[inline]
    pub fn contains_point(&self, value: T) -> bool {
        self.start_inclusive <= value && value < self.end_exclusive
    }

    /// Whether the bounds of `other` lie within those of `self`; an interval contains itself.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// assert!(a.contains_interval(ClosedOpenInterval::new(2, 8)));
    /// assert!(a.contains_interval(a));
    /// ```
    #[inline]
    pub fn contains_interval(&self, other: Self) -> bool {
        self.start_inclusive <= other.start_inclusive && other.end_exclusive <= self.end_exclusive
    }

    /// The number of integers in the interval, `end - start`.
    ///
    /// # Panics
    ///
    /// Panics if the length exceeds the greatest value of `T`, which only an interval of a signed type wider
    /// than half its range has.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// assert_eq!(ClosedOpenInterval::new(10, 20).len(), 10);
    /// ```
    #[inline]
    #[track_caller]
    pub fn len(&self) -> T {
        match self.end_exclusive.checked_sub(&self.start_inclusive) {
            Some(length) => length,
            None => panic_length_overflow(),
        }
    }

    /// Whether the interval holds no integer (`start == end`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// assert!(ClosedOpenInterval::new(10, 10).is_empty());
    /// assert!(!ClosedOpenInterval::new(10, 11).is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start_inclusive == self.end_exclusive
    }

    /// The integers both intervals hold, or `None` if they share none.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// assert_eq!(a.intersection(ClosedOpenInterval::new(5, 15)), Some(ClosedOpenInterval::new(5, 10)));
    /// assert_eq!(a.intersection(ClosedOpenInterval::new(10, 15)), None);
    /// ```
    #[inline]
    pub fn intersection(&self, other: Self) -> Option<Self> {
        let start = max(self.start_inclusive, other.start_inclusive);
        let end = min(self.end_exclusive, other.end_exclusive);
        (start < end).then(|| Self::from_ordered(start, end))
    }

    /// The interval from the lesser start to the greater end, or `None` if a gap separates the two.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 10);
    /// assert_eq!(a.union(ClosedOpenInterval::new(10, 20)), Some(ClosedOpenInterval::new(0, 20)));
    /// assert_eq!(a.union(ClosedOpenInterval::new(11, 20)), None);
    /// ```
    #[inline]
    pub fn union(&self, other: Self) -> Option<Self> {
        self.intersects_or_adjacent(other).then(|| {
            Self::from_ordered(
                min(self.start_inclusive, other.start_inclusive),
                max(self.end_exclusive, other.end_exclusive),
            )
        })
    }

    /// The mean of the bounds, rounded down: `floor((start + end) / 2)`, for every interval of `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// assert_eq!(ClosedOpenInterval::new(0, 10).midpoint(), 5);
    /// assert_eq!(ClosedOpenInterval::new(-3, 0).midpoint(), -2);
    /// ```
    #[inline]
    pub fn midpoint(&self) -> T {
        // The shared bits plus half of the differing ones is the floor of the mean, and neither sum can
        // leave `T`: `>>` on a signed integer is arithmetic.
        let (start, end) = (self.start_inclusive, self.end_exclusive);
        (start & end) + ((start ^ end) >> 1)
    }

    /// The integers strictly between the two intervals, or `None` if they intersect or touch.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let a = ClosedOpenInterval::new(0, 5);
    /// let b = ClosedOpenInterval::new(10, 15);
    /// assert_eq!(a.gap(b), Some(ClosedOpenInterval::new(5, 10)));
    /// assert_eq!(b.gap(a), Some(ClosedOpenInterval::new(5, 10)));
    /// ```
    #[inline]
    pub fn gap(&self, other: Self) -> Option<Self> {
        if self.end_exclusive < other.start_inclusive {
            Some(Self::from_ordered(
                self.end_exclusive,
                other.start_inclusive,
            ))
        } else if other.end_exclusive < self.start_inclusive {
            Some(Self::from_ordered(
                other.end_exclusive,
                self.start_inclusive,
            ))
        } else {
            None
        }
    }

    /// The interval cut at `value` into `[start, value)` and `[value, end)`, or `None` unless
    /// `start < value < end`, so that both parts hold an integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let interval = ClosedOpenInterval::new(0, 10);
    /// assert_eq!(
    ///     interval.split_at(4),
    ///     Some((ClosedOpenInterval::new(0, 4), ClosedOpenInterval::new(4, 10)))
    /// );
    /// assert_eq!(interval.split_at(0), None);
    /// ```
    #[inline]
    pub fn split_at(&self, value: T) -> Option<(Self, Self)> {
        (self.start_inclusive < value && value < self.end_exclusive).then(|| {
            (
                Self::from_ordered(self.start_inclusive, value),
                Self::from_ordered(value, self.end_exclusive),
            )
        })
    }

    /// The integers of the interval, in increasing order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use talos_core::ClosedOpenInterval;
    ///
    /// let points: Vec<_> = ClosedOpenInterval::new(1, 4).iter().collect();
    /// assert_eq!(points, [1, 2, 3]);
    /// ```
    #[inline]
    pub fn iter(&self) -> ClosedOpenIntervalIter<T> {
        ClosedOpenIntervalIter {
            front: self.start_inclusive,
            back: self.end_exclusive,
        }
    }
}

/// Panics with the message of [`ClosedOpenInterval::new`] for a start after the end.
// Kept out of line so that the check in `new` stays a compare and a branch on the caller's path.
#[cold]
#[inline(never)]
#[track_caller]
fn panic_start_after_end<T: fmt::Debug>(start_inclusive: T, end_exclusive: T) -> ! {
    panic!(
        "called `ClosedOpenInterval::new` with start {start_inclusive:?} after end {end_exclusive:?}"
    )
}

/// Panics with the message of [`ClosedOpenInterval::len`] for a length `T` cannot hold.
// Kept out of line so that the check in `len` stays a compare and a branch on the caller's path.
#[cold]
#[inline(never)]
#[track_caller]
fn panic_length_overflow() -> ! {
    panic!("called `ClosedOpenInterval::len` on an interval longer than its type can count")
}

impl<T> Default for ClosedOpenInterval<T>
where
    T: PrimInt,
{
    /// The empty interval at zero, `[0, 0)`.
    #[inline]
    fn default() -> Self {
        Self::from_ordered(T::zero(), T::zero())
    }
}

impl<T> fmt::Display for ClosedOpenInterval<T>
where
    T: PrimInt + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.start_inclusive, self.end_exclusive)
    }
}

impl<T> RangeBounds<T> for ClosedOpenInterval<T>
where
    T: PrimInt,
{
    fn start_bound(&self) -> Bound<&T> {
        Bound::Included(&self.start_inclusive)
    }

    fn end_bound(&self) -> Bound<&T> {
        Bound::Excluded(&self.end_exclusive)
    }
}

impl<T> IntoIterator for ClosedOpenInterval<T>
where
    T: PrimInt,
{
    type Item = T;
    type IntoIter = ClosedOpenIntervalIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> IntoIterator for &ClosedOpenInterval<T>
where
    T: PrimInt,
{
    type Item = T;
    type IntoIter = ClosedOpenIntervalIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> From<Range<T>> for ClosedOpenInterval<T>
where
    T: PrimInt + fmt::Debug,
{
    /// The interval of the same integers as `range`.
    ///
    /// # Panics
    ///
    /// Panics if the range starts after its end, as [`ClosedOpenInterval::new`].
    #[inline]
    #[track_caller]
    fn from(range: Range<T>) -> Self {
        Self::new(range.start, range.end)
    }
}

impl<T> From<ClosedOpenInterval<T>> for Range<T>
where
    T: PrimInt,
{
    #[inline]
    fn from(interval: ClosedOpenInterval<T>) -> Self {
        interval.start_inclusive..interval.end_exclusive
    }
}

/// The integers of a [`ClosedOpenInterval`], from either end.
///
/// The integers not yet yielded are always `[front, back)`.
#[derive(Clone, Debug)]
pub struct ClosedOpenIntervalIter<T>
where
    T: PrimInt,
{
    front: T,
    back: T,
}

impl<T> Iterator for ClosedOpenIntervalIter<T>
where
    T: PrimInt,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        // `front < back` keeps `front + 1` within `T`.
        (self.front < self.back).then(|| {
            let point = self.front;
            self.front = point + T::one();
            point
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // A count that `T` or `usize` cannot hold is only bounded below, as for std's ranges.
        match self
            .back
            .checked_sub(&self.front)
            .and_then(|count| count.to_usize())
        {
            Some(count) => (count, Some(count)),
            None => (usize::MAX, None),
        }
    }
}

impl<T> DoubleEndedIterator for ClosedOpenIntervalIter<T>
where
    T: PrimInt,
{
    fn next_back(&mut self) -> Option<T> {
        // `front < back` keeps `back - 1` within `T`.
        (self.front < self.back).then(|| {
            self.back = self.back - T::one();
            self.back
        })
    }
}

impl<T> FusedIterator for ClosedOpenIntervalIter<T> where T: PrimInt {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    fn interval(start: i64, end: i64) -> ClosedOpenInterval<i64> {
        ClosedOpenInterval::new(start, end)
    }

    #[test]
    fn test_new_keeps_its_bounds() {
        let a = interval(10, 20);
        assert_eq!((a.start(), a.end(), a.len()), (10, 20, 10));
        assert!(!a.is_empty());
        assert!(interval(10, 10).is_empty());
    }

    #[test]
    fn test_try_new_refuses_a_start_after_the_end() {
        assert_eq!(ClosedOpenInterval::try_new(5, 10), Some(interval(5, 10)));
        assert_eq!(ClosedOpenInterval::try_new(5, 5), Some(interval(5, 5)));
        assert_eq!(ClosedOpenInterval::try_new(10, 5), None);
    }

    #[test]
    #[should_panic(expected = "called `ClosedOpenInterval::new` with start 10 after end 5")]
    fn test_new_panics_on_a_start_after_the_end() {
        let _ = interval(10, 5);
    }

    #[test]
    #[should_panic(
        expected = "called `ClosedOpenInterval::len` on an interval longer than its type can count"
    )]
    fn test_len_panics_when_the_type_cannot_count_it() {
        let _ = interval(i64::MIN, 1).len();
    }

    #[test]
    fn test_default_is_empty_at_zero() {
        assert_eq!(
            ClosedOpenInterval::<i32>::default(),
            ClosedOpenInterval::new(0, 0)
        );
    }

    #[test]
    fn test_intersects() {
        let a = interval(0, 10);
        assert!(!a.intersects(interval(-10, -5)));
        assert!(!a.intersects(interval(-5, 0)));
        assert!(a.intersects(interval(-5, 5)));
        assert!(a.intersects(interval(2, 8)));
        assert!(a.intersects(a));
        assert!(a.intersects(interval(5, 15)));
        assert!(!a.intersects(interval(10, 15)));
        assert!(!a.intersects(interval(11, 15)));
    }

    #[test]
    fn test_an_empty_interval_intersects_nothing() {
        let (empty, around) = (interval(0, 0), interval(-1, 1));
        assert!(!empty.intersects(around));
        assert!(!around.intersects(empty));
        assert_eq!(empty.intersection(around), None);
    }

    #[test]
    fn test_adjacent() {
        let a = interval(0, 10);
        assert!(a.adjacent(interval(-5, 0)));
        assert!(a.adjacent(interval(10, 15)));
        assert!(!a.adjacent(interval(9, 11)));
        assert!(!a.adjacent(interval(12, 15)));
    }

    #[test]
    fn test_disjoint_and_intersects_or_adjacent_are_opposites() {
        let a = interval(0, 10);
        for other in [interval(5, 15), interval(10, 20), interval(11, 20)] {
            assert_eq!(a.disjoint(other), !a.intersects_or_adjacent(other));
        }
        assert!(a.disjoint(interval(11, 20)));
        assert!(!a.disjoint(interval(10, 20)));
    }

    #[test]
    fn test_contains_point() {
        let a = interval(0, 10);
        assert!(a.contains_point(0));
        assert!(a.contains_point(9));
        assert!(!a.contains_point(10));
        assert!(!a.contains_point(-1));
    }

    #[test]
    fn test_contains_interval() {
        let a = interval(0, 10);
        assert!(a.contains_interval(interval(0, 10)));
        assert!(a.contains_interval(interval(2, 8)));
        assert!(!a.contains_interval(interval(-1, 5)));
        assert!(!a.contains_interval(interval(5, 11)));
        assert!(!a.contains_interval(interval(20, 30)));
    }

    #[test]
    fn test_intersection() {
        let a = interval(0, 10);
        assert_eq!(a.intersection(interval(5, 15)), Some(interval(5, 10)));
        assert_eq!(a.intersection(interval(2, 8)), Some(interval(2, 8)));
        assert_eq!(a.intersection(interval(10, 20)), None);
        assert_eq!(a.intersection(interval(12, 20)), None);
    }

    #[test]
    fn test_union() {
        let a = interval(0, 10);
        assert_eq!(a.union(interval(5, 15)), Some(interval(0, 15)));
        assert_eq!(a.union(interval(10, 20)), Some(interval(0, 20)));
        assert_eq!(a.union(interval(2, 8)), Some(a));
        assert_eq!(a.union(interval(12, 20)), None);
    }

    #[test]
    fn test_gap() {
        let (a, b) = (interval(0, 5), interval(10, 15));
        assert_eq!(a.gap(b), Some(interval(5, 10)));
        assert_eq!(b.gap(a), Some(interval(5, 10)));
        assert_eq!(a.gap(interval(5, 10)), None);
        assert_eq!(a.gap(interval(4, 6)), None);
    }

    #[test]
    fn test_midpoint() {
        assert_eq!(interval(0, 10).midpoint(), 5);
        assert_eq!(interval(0, 3).midpoint(), 1);
        assert_eq!(interval(-10, -4).midpoint(), -7);
        assert_eq!(interval(-3, 0).midpoint(), -2);
        assert_eq!(ClosedOpenInterval::<u8>::new(250, 254).midpoint(), 252);
    }

    #[test]
    fn test_midpoint_of_the_widest_interval_does_not_overflow() {
        assert_eq!(interval(i64::MIN, i64::MAX).midpoint(), -1);
    }

    #[test]
    fn test_split_at() {
        let a = interval(0, 10);
        assert_eq!(a.split_at(4), Some((interval(0, 4), interval(4, 10))));
        assert_eq!(a.split_at(0), None);
        assert_eq!(a.split_at(10), None);
    }

    #[test]
    fn test_iterator_yields_from_both_ends() {
        let mut points = interval(1, 4).iter();
        assert_eq!(points.next(), Some(1));
        assert_eq!(points.next_back(), Some(3));
        assert_eq!(points.next(), Some(2));
        assert_eq!(points.next(), None);
        assert_eq!(points.next_back(), None);
    }

    #[test]
    fn test_iterator_of_an_empty_interval_is_empty() {
        assert_eq!(interval(5, 5).iter().next(), None);
    }

    #[test]
    fn test_iterator_size_hint_counts_the_points_left() {
        let mut points = interval(1, 4).iter();
        assert_eq!(points.size_hint(), (3, Some(3)));
        points.next_back();
        assert_eq!(points.size_hint(), (2, Some(2)));
    }

    #[test]
    fn test_iterator_size_hint_of_an_uncountable_interval_is_unbounded() {
        assert_eq!(
            interval(i64::MIN, i64::MAX).iter().size_hint(),
            (usize::MAX, None)
        );
    }

    #[test]
    fn test_into_iterator_by_value_and_by_reference() {
        let a = interval(0, 3);
        assert_eq!((&a).into_iter().collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(a.into_iter().collect::<Vec<_>>(), [0, 1, 2]);
    }

    #[test]
    fn test_display_and_debug() {
        let a = interval(10, 20);
        assert_eq!(format!("{a}"), "[10, 20)");
        assert_eq!(
            format!("{a:?}"),
            "ClosedOpenInterval { start_inclusive: 10, end_exclusive: 20 }"
        );
    }

    #[test]
    fn test_conversions_with_range() {
        assert_eq!(ClosedOpenInterval::from(0..10), interval(0, 10));
        assert_eq!(Range::from(interval(0, 10)), 0..10);
    }

    #[test]
    fn test_range_bounds() {
        let a = interval(5, 10);
        assert_eq!(a.start_bound(), Bound::Included(&5));
        assert_eq!(a.end_bound(), Bound::Excluded(&10));
    }

    // The integers of `[start, end)` as a set: the model the interval's set operations are checked against.
    fn points(interval: ClosedOpenInterval<i64>) -> BTreeSet<i64> {
        (interval.start()..interval.end()).collect()
    }

    fn small_interval() -> impl Strategy<Value = ClosedOpenInterval<i64>> {
        (-20i64..20, 0i64..20).prop_map(|(start, length)| interval(start, start + length))
    }

    proptest! {
        #[test]
        fn test_try_new_is_some_exactly_for_ordered_bounds(start in -50i64..50, end in -50i64..50) {
            prop_assert_eq!(ClosedOpenInterval::try_new(start, end).is_some(), start <= end);
        }

        #[test]
        fn test_intersects_and_intersection_match_the_point_sets(
            a in small_interval(),
            b in small_interval(),
        ) {
            let shared: BTreeSet<i64> = points(a).intersection(&points(b)).copied().collect();
            prop_assert_eq!(a.intersects(b), !shared.is_empty());
            prop_assert_eq!(a.intersection(b).map(points).unwrap_or_default(), shared);
        }

        #[test]
        fn test_union_of_overlapping_or_touching_intervals_covers_both(
            a in small_interval(),
            b in small_interval(),
        ) {
            match a.union(b) {
                Some(union) => {
                    let both: BTreeSet<i64> = points(a).union(&points(b)).copied().collect();
                    prop_assert!(both.is_subset(&points(union)));
                    prop_assert_eq!(union.start(), a.start().min(b.start()));
                    prop_assert_eq!(union.end(), a.end().max(b.end()));
                }
                None => prop_assert!(a.end() < b.start() || b.end() < a.start()),
            }
        }

        #[test]
        fn test_contains_point_matches_the_point_set(a in small_interval(), value in -50i64..50) {
            prop_assert_eq!(a.contains_point(value), points(a).contains(&value));
        }

        #[test]
        fn test_iterator_yields_the_point_set_in_order(a in small_interval()) {
            prop_assert_eq!(a.iter().collect::<Vec<_>>(), points(a).into_iter().collect::<Vec<_>>());
            prop_assert_eq!(a.iter().rev().collect::<Vec<_>>(), points(a).into_iter().rev().collect::<Vec<_>>());
            prop_assert_eq!(a.iter().size_hint().1, Some(points(a).len()));
        }

        #[test]
        fn test_midpoint_is_the_floor_of_the_mean(a in any::<i32>(), b in any::<i32>()) {
            // In `i64` the sum of two `i32` cannot overflow, so the plain mean is the reference.
            let narrow = ClosedOpenInterval::new(a.min(b), a.max(b));
            let mean = (i64::from(a) + i64::from(b)).div_euclid(2);
            prop_assert_eq!(i64::from(narrow.midpoint()), mean);
        }
    }
}
