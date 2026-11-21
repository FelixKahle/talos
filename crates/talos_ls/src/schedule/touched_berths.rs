// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The berths whose order a sequence of moves changed, which the decoder decodes again.

use std::fmt;
use std::iter::FusedIterator;
use talos_model::BerthIndex;
use typed_index_collections::TiVec;

/// The berths a sequence of moves touched, as a set of berths.
///
/// Marking a berth, asking whether one is marked, and walking the marked ones cost what was marked, not the count
/// of berths, and marking never allocates: the set holds room for every berth from the start.
// Invariant: `touched` holds every berth whose mark is set, once each, in the order they were first marked, and its
// capacity is at least the count of berths, the length of `marked`. Every entry of `touched` is below that count,
// which is what the decoder's unchecked reads through the walk rest on.
#[derive(Debug)]
pub struct TouchedBerths {
    /// Whether each berth is marked, per berth.
    marked: TiVec<BerthIndex, bool>,
    /// The marked berths, in the order of their first mark.
    touched: Vec<BerthIndex>,
}

impl TouchedBerths {
    /// The empty set of `num_berths` berths.
    #[inline]
    pub fn new(num_berths: usize) -> Self {
        Self {
            marked: TiVec::from(vec![false; num_berths]),
            touched: Vec::with_capacity(num_berths),
        }
    }

    /// Makes the set the empty set of `num_berths` berths, reusing its memory.
    #[inline]
    pub fn resize(&mut self, num_berths: usize) {
        self.reset();
        self.marked.resize(num_berths, false);
        self.touched.reserve(num_berths);
    }

    /// The number of berths.
    #[inline(always)]
    pub fn num_berths(&self) -> usize {
        self.marked.len()
    }

    /// Marks `berth`; nothing changes if it is marked already.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not below the number of berths.
    #[inline]
    #[track_caller]
    pub fn touch(&mut self, berth: BerthIndex) {
        assert!(
            berth.index() < self.num_berths(),
            "called `TouchedBerths::touch` with {berth}, outside a set of {} berths",
            self.num_berths()
        );
        // SAFETY: `berth` is below the number of berths, as the check above found.
        unsafe { self.touch_unchecked(berth) }
    }

    /// Marks `berth`, for a berth the caller knows to be below the number of berths.
    ///
    /// # Safety
    ///
    /// `berth` must be below the number of berths, the condition [`touch`](Self::touch) checks.
    #[inline(always)]
    pub unsafe fn touch_unchecked(&mut self, berth: BerthIndex) {
        debug_assert!(
            berth.index() < self.num_berths(),
            "called `TouchedBerths::touch_unchecked` with {berth}, outside a set of {} berths",
            self.num_berths()
        );
        // SAFETY: `berth` is below the number of berths, the precondition of this function.
        let mark = unsafe { self.marked.get_unchecked_mut(berth) };
        if !*mark {
            *mark = true;
            // Within the capacity, which counts every berth, so the push does not allocate.
            self.touched.push(berth);
        }
    }

    /// Whether `berth` is marked.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not below the number of berths.
    #[inline]
    #[track_caller]
    pub fn is_touched(&self, berth: BerthIndex) -> bool {
        assert!(
            berth.index() < self.num_berths(),
            "called `TouchedBerths::is_touched` with {berth}, outside a set of {} berths",
            self.num_berths()
        );
        self.marked[berth]
    }

    /// Unmarks every berth, at a cost of the marked ones.
    #[inline]
    pub fn reset(&mut self) {
        for &berth in &self.touched {
            debug_assert!(
                berth.index() < self.marked.len(),
                "a marked berth should lie below the number of berths"
            );
            // SAFETY: every marked berth is below the number of berths by the invariant of `TouchedBerths`.
            *unsafe { self.marked.get_unchecked_mut(berth) } = false;
        }
        self.touched.clear();
    }

    /// The marked berths, each once, in the order of their first mark.
    #[inline(always)]
    pub fn iter_touched_berths(&self) -> TouchedBerthsIter<'_> {
        TouchedBerthsIter {
            inner: self.touched.iter(),
        }
    }
}

// Written out so that two sets of the same berths are equal, whatever order their berths were first marked in.
impl PartialEq for TouchedBerths {
    fn eq(&self, other: &Self) -> bool {
        self.marked == other.marked
    }
}

impl Eq for TouchedBerths {}

// Written out so that the copy, too, holds room for every berth.
impl Clone for TouchedBerths {
    fn clone(&self) -> Self {
        let mut touched = Vec::with_capacity(self.num_berths());
        touched.extend_from_slice(&self.touched);
        Self {
            marked: self.marked.clone(),
            touched,
        }
    }
}

impl fmt::Display for TouchedBerths {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "touched:")?;
        for berth in self.iter_touched_berths() {
            write!(f, " {berth}")?;
        }
        Ok(())
    }
}

impl<'a> IntoIterator for &'a TouchedBerths {
    type Item = BerthIndex;
    type IntoIter = TouchedBerthsIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_touched_berths()
    }
}

/// The marked berths of a [`TouchedBerths`], in the order of their first mark.
#[derive(Clone, Debug)]
pub struct TouchedBerthsIter<'a> {
    /// The walk over the list of marked berths.
    inner: std::slice::Iter<'a, BerthIndex>,
}

impl Iterator for TouchedBerthsIter<'_> {
    type Item = BerthIndex;

    #[inline(always)]
    fn next(&mut self) -> Option<BerthIndex> {
        self.inner.next().copied()
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for TouchedBerthsIter<'_> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<BerthIndex> {
        self.inner.next_back().copied()
    }
}

impl ExactSizeIterator for TouchedBerthsIter<'_> {}

impl FusedIterator for TouchedBerthsIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn berth(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    #[test]
    fn test_a_berth_touched_twice_is_walked_once_in_the_order_of_its_first_touch() {
        let mut touched = TouchedBerths::new(4);
        touched.touch(berth(2));
        touched.touch(berth(0));
        touched.touch(berth(2));
        assert_eq!(
            touched.iter_touched_berths().collect::<Vec<_>>(),
            [berth(2), berth(0)]
        );
        assert!(touched.is_touched(berth(0)) && !touched.is_touched(berth(1)));
    }

    #[test]
    fn test_reset_unmarks_every_berth() {
        let mut touched = TouchedBerths::new(3);
        touched.touch(berth(1));
        touched.reset();
        assert_eq!(touched.iter_touched_berths().len(), 0);
        assert!(!touched.is_touched(berth(1)));
    }

    #[test]
    fn test_resize_empties_the_set_and_counts_the_new_berths() {
        let mut touched = TouchedBerths::new(2);
        touched.touch(berth(1));
        touched.resize(5);
        assert_eq!(touched.num_berths(), 5);
        assert_eq!(touched.iter_touched_berths().len(), 0);
        touched.touch(berth(4));
        assert!(touched.is_touched(berth(4)));
    }

    #[test]
    fn test_sets_of_the_same_berths_are_equal_whatever_order_they_were_touched_in() {
        let (mut one, mut other) = (TouchedBerths::new(4), TouchedBerths::new(4));
        one.touch(berth(1));
        one.touch(berth(3));
        other.touch(berth(3));
        other.touch(berth(1));
        assert_eq!(one, other);
        other.touch(berth(0));
        assert_ne!(one, other);
        assert_ne!(TouchedBerths::new(3), TouchedBerths::new(4));
    }

    #[test]
    #[should_panic(
        expected = "called `TouchedBerths::is_touched` with Berth 3, outside a set of 3 berths"
    )]
    fn test_is_touched_refuses_a_berth_outside() {
        let _ = TouchedBerths::new(3).is_touched(berth(3));
    }

    #[test]
    #[should_panic(
        expected = "called `TouchedBerths::touch` with Berth 3, outside a set of 3 berths"
    )]
    fn test_touch_refuses_a_berth_outside() {
        TouchedBerths::new(3).touch(berth(3));
    }

    proptest! {
        #[test]
        fn test_touch_and_is_touched_refuse_every_berth_outside(
            num_berths in 0_usize..8,
            beyond in 0_usize..64,
        ) {
            let outside = berth(num_berths + beyond);
            let mut touched = TouchedBerths::new(num_berths);
            let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| touched.touch(outside)));
            prop_assert!(refused.is_err());
            prop_assert_eq!(touched.iter_touched_berths().len(), 0);
            let refused = std::panic::catch_unwind(|| TouchedBerths::new(num_berths).is_touched(outside));
            prop_assert!(refused.is_err());
        }

        #[test]
        fn test_marks_match_a_set_and_touch_matches_its_unchecked_twin(
            num_berths in 1_usize..8,
            marks in proptest::collection::vec(0_usize..8, 0..20),
        ) {
            let mut checked = TouchedBerths::new(num_berths);
            let mut unchecked = TouchedBerths::new(num_berths);
            let mut reference = std::collections::BTreeSet::new();
            for &mark in marks.iter().filter(|&&mark| mark < num_berths) {
                checked.touch(berth(mark));
                // SAFETY: `mark` is below the number of berths, as the filter keeps.
                unsafe { unchecked.touch_unchecked(berth(mark)) };
                reference.insert(mark);
            }
            prop_assert_eq!(&checked, &unchecked);
            let walked: std::collections::BTreeSet<_> = checked.iter_touched_berths().map(BerthIndex::index).collect();
            prop_assert_eq!(walked.len(), checked.iter_touched_berths().len());
            prop_assert_eq!(&walked, &reference);
            for index in 0..num_berths {
                prop_assert_eq!(checked.is_touched(berth(index)), reference.contains(&index));
            }
        }
    }
}
