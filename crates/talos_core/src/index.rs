// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed ids: one type per kind of id, so that an id of one table cannot index another.

use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

// An id stores a `u32` and answers a `usize`; that conversion is exact only where a `usize` holds every `u32`.
const _: () = assert!(
    usize::BITS >= u32::BITS,
    "a `usize` should hold every `u32`"
);

/// The marker that makes a [`TypedIndex`] an id of one kind.
///
/// A tag is a zero-sized type that exists only to tell one kind of id from another. Each kind of id declares
/// its own tag and names its id type `TypedIndex<ThatTag>`.
pub trait TypedIndexTag: Copy + Eq + Ord + Hash + fmt::Debug {
    /// The name of the id kind, as `Debug` and `Display` print it (`Node 3`).
    const NAME: &'static str;
}

/// An id of the kind that `T` tags: a position in the tables of that kind.
///
/// Ids of different kinds are different types, so an id of one table cannot be handed to another by mistake.
/// The id stores a `u32`, which halves the memory of every stored column of ids, and speaks `usize` at its
/// interface: it is made from a `usize` and answers one. An id is so at most `u32::MAX`, which [`new`] checks.
///
/// [`new`]: TypedIndex::new
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypedIndex<T>
where
    T: TypedIndexTag,
{
    index: u32,
    tag: PhantomData<T>,
}

impl<T> TypedIndex<T>
where
    T: TypedIndexTag,
{
    /// The id at position `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than `u32::MAX`.
    #[inline]
    #[track_caller]
    pub const fn new(index: usize) -> Self {
        match Self::try_new(index) {
            Some(id) => id,
            None => panic!("called `TypedIndex::new` with an index beyond u32::MAX"),
        }
    }

    /// The id at position `index`, or `None` if `index` is greater than `u32::MAX`.
    #[inline]
    pub const fn try_new(index: usize) -> Option<Self> {
        // Both casts are exact: a `usize` holds every `u32`, as the assertion at the top of this module states,
        // and `index` is at most `u32::MAX` once past the check.
        if index > u32::MAX as usize {
            return None;
        }
        Some(Self {
            index: index as u32,
            tag: PhantomData,
        })
    }

    /// The position this id stands for.
    #[inline]
    pub const fn index(self) -> usize {
        // Exact: a `usize` holds every `u32`, as the assertion at the top of this module states.
        self.index as usize
    }

    /// The id of kind `U` at the same position.
    ///
    /// This is for tables of two kinds whose positions correspond by construction, such as the first nodes of a
    /// graph that stand for the vessels of a problem.
    #[inline]
    pub const fn map<U>(self) -> TypedIndex<U>
    where
        U: TypedIndexTag,
    {
        TypedIndex {
            index: self.index,
            tag: PhantomData,
        }
    }
}

impl<T> fmt::Debug for TypedIndex<T>
where
    T: TypedIndexTag,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", T::NAME, self.index)
    }
}

impl<T> fmt::Display for TypedIndex<T>
where
    T: TypedIndexTag,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", T::NAME, self.index)
    }
}

impl<T> From<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    /// The id at position `index`, as [`TypedIndex::new`].
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than `u32::MAX`.
    #[inline]
    #[track_caller]
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

impl<T> From<TypedIndex<T>> for usize
where
    T: TypedIndexTag,
{
    #[inline]
    fn from(id: TypedIndex<T>) -> Self {
        id.index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    struct TestTag;

    impl TypedIndexTag for TestTag {
        const NAME: &'static str = "Test";
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    struct OtherTag;

    impl TypedIndexTag for OtherTag {
        const NAME: &'static str = "Other";
    }

    type TestIndex = TypedIndex<TestTag>;

    // Every position beyond `u32::MAX` that a 64-bit `usize` can name.
    #[cfg(target_pointer_width = "64")]
    const FIRST_BEYOND_U32: usize = 1 << 32;

    #[test]
    fn test_index_answers_the_position_it_was_made_from() {
        assert_eq!(TestIndex::new(10).index(), 10);
        assert_eq!(TestIndex::new(0).index(), 0);
    }

    fn largest_u32() -> usize {
        usize::try_from(u32::MAX).expect("a u32 should fit a usize")
    }

    #[test]
    fn test_try_new_accepts_u32_max() {
        let largest = largest_u32();
        assert_eq!(
            TestIndex::try_new(largest).map(TestIndex::index),
            Some(largest)
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_try_new_refuses_beyond_u32_max() {
        assert_eq!(TestIndex::try_new(FIRST_BEYOND_U32), None);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "called `TypedIndex::new` with an index beyond u32::MAX")]
    fn test_new_panics_beyond_u32_max() {
        let _ = TestIndex::new(FIRST_BEYOND_U32);
    }

    #[test]
    fn test_conversions_with_usize_round_trip() {
        let id: TestIndex = 42.into();
        assert_eq!(id, TestIndex::new(42));
        assert_eq!(usize::from(id), 42);
    }

    #[test]
    fn test_map_keeps_the_position() {
        let other: TypedIndex<OtherTag> = TestIndex::new(7).map();
        assert_eq!(other.index(), 7);
        assert_eq!(format!("{other}"), "Other 7");
    }

    #[test]
    fn test_debug_and_display_name_the_kind() {
        let id = TestIndex::new(7);
        assert_eq!(format!("{id}"), "Test 7");
        assert_eq!(format!("{id:?}"), "Test 7");
    }

    #[test]
    fn test_ids_order_by_position() {
        assert!(TestIndex::new(1) < TestIndex::new(2));
        assert_ne!(TestIndex::new(1), TestIndex::new(2));
    }

    proptest! {
        #[test]
        fn test_try_new_is_some_exactly_up_to_u32_max(index in prop_oneof![0..=largest_u32(), any::<usize>()]) {
            let made = TestIndex::try_new(index);
            prop_assert_eq!(made.is_some(), u32::try_from(index).is_ok());
            if let Some(id) = made {
                prop_assert_eq!(id.index(), index);
            }
        }

        #[test]
        fn test_order_of_ids_is_the_order_of_positions(a in any::<u32>(), b in any::<u32>()) {
            let (a, b) = (usize::try_from(a).unwrap(), usize::try_from(b).unwrap());
            prop_assert_eq!(TestIndex::new(a).cmp(&TestIndex::new(b)), a.cmp(&b));
        }
    }
}
