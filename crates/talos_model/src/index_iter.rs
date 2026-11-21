// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The walk over every id of one kind.

use derive_where::derive_where;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ops::Range;
use talos_core::{TypedIndex, TypedIndexTag};

/// The ids of one kind from the first up to a count, in increasing order.
///
/// This is the walk over every id of a kind up to a count, such as every vessel or every berth of a problem.
// The positions are kept as `u32`, the width of an id, so that making an id of each is exact without a check.
#[derive_where(Clone, Debug, PartialEq, Eq)]
pub struct TypedIndexIter<T>
where
    T: TypedIndexTag,
{
    positions: Range<u32>,
    tag: PhantomData<T>,
}

impl<T> TypedIndexIter<T>
where
    T: TypedIndexTag,
{
    /// The ids at the positions `0..count`.
    ///
    /// # Panics
    ///
    /// Panics if `count` is greater than `u32::MAX`.
    #[inline]
    #[track_caller]
    pub fn new(count: usize) -> Self {
        match u32::try_from(count) {
            Ok(end) => Self {
                positions: 0..end,
                tag: PhantomData,
            },
            Err(_) => panic!("called `TypedIndexIter::new` with count {count}, beyond u32::MAX"),
        }
    }
}

/// The id at `position`, which as a `u32` every id can name.
#[inline]
fn id_at<T>(position: u32) -> TypedIndex<T>
where
    T: TypedIndexTag,
{
    let position = usize::try_from(position).expect("a usize should hold every u32");
    TypedIndex::new(position)
}

impl<T> Iterator for TypedIndexIter<T>
where
    T: TypedIndexTag,
{
    type Item = TypedIndex<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.positions.next().map(id_at)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.positions.size_hint()
    }
}

impl<T> DoubleEndedIterator for TypedIndexIter<T>
where
    T: TypedIndexTag,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.positions.next_back().map(id_at)
    }
}

impl<T> ExactSizeIterator for TypedIndexIter<T> where T: TypedIndexTag {}

impl<T> FusedIterator for TypedIndexIter<T> where T: TypedIndexTag {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Debug)]
    struct TestTag;

    impl TypedIndexTag for TestTag {
        const NAME: &'static str = "Test";
    }

    type TestIndex = TypedIndex<TestTag>;
    type TestIndexIter = TypedIndexIter<TestTag>;

    #[test]
    fn test_iter_of_zero_ids_is_empty() {
        assert_eq!(TestIndexIter::new(0).next(), None);
    }

    #[test]
    fn test_iter_walks_both_ends_towards_each_other() {
        let mut ids = TestIndexIter::new(3);
        assert_eq!(ids.next(), Some(TestIndex::new(0)));
        assert_eq!(ids.next_back(), Some(TestIndex::new(2)));
        assert_eq!(ids.len(), 1);
        assert_eq!(ids.next(), Some(TestIndex::new(1)));
        assert_eq!(ids.next_back(), None);
        assert_eq!(ids.next(), None);
    }

    #[test]
    fn test_iter_accepts_u32_max() {
        let largest = usize::try_from(u32::MAX).expect("a usize should hold every u32");
        let mut ids = TestIndexIter::new(largest);
        assert_eq!(ids.len(), largest);
        assert_eq!(ids.next_back(), Some(TestIndex::new(largest - 1)));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(
        expected = "called `TypedIndexIter::new` with count 4294967296, beyond u32::MAX"
    )]
    fn test_iter_refuses_a_count_beyond_u32_max() {
        let _ = TestIndexIter::new(1 << 32);
    }

    proptest! {
        #[test]
        fn test_iter_yields_each_position_below_the_count_once_in_order(count in 0usize..200) {
            let positions: Vec<usize> = TestIndexIter::new(count).map(TestIndex::index).collect();
            prop_assert_eq!(positions, (0..count).collect::<Vec<_>>());
        }

        #[test]
        fn test_iter_reports_the_ids_it_has_left(count in 0usize..200, taken in 0usize..200) {
            let mut ids = TestIndexIter::new(count);
            for _ in 0..taken {
                ids.next();
            }
            let left = count.saturating_sub(taken);
            prop_assert_eq!(ids.size_hint(), (left, Some(left)));
            prop_assert_eq!(ids.count(), left);
        }
    }
}
