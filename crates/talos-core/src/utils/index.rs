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

//! # Strongly Typed Indices (Zero-Cost)
//!
//! Phantom-typed wrappers around `usize` to prevent mixing indices from
//! different domains (e.g., vessels vs. berths). `TypedIndex<T>` carries a
//! tag type `T: TypedIndexTag` that encodes intent at the type level, while
//! compiling down to a transparent `usize` (no runtime overhead).
//!
//! ## Motivation
//!
//! In large scheduling and optimization pipelines, multiple index spaces are
//! used concurrently. Raw `usize` invites accidental swaps and hard-to-trace
//! bugs. Phantom-tagged indices provide compile-time guarantees with minimal
//! ceremony and excellent ergonomics.

/// A trait to tag typed indices with a name for debugging and display purposes.
pub trait TypedIndexTag:
    Copy + Clone + PartialEq + Eq + PartialOrd + Ord + std::hash::Hash
{
    const NAME: &'static str;
}

/// A strongly typed index that is associated with a specific tag type `T`.
///
/// This struct wraps a `usize` index and uses a phantom type parameter `T`
/// to provide type safety and prevent mixing indices of different types.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypedIndex<T>
where
    T: TypedIndexTag,
{
    index: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> TypedIndex<T>
where
    T: TypedIndexTag,
{
    /// Creates a new `TypedIndex` with the given `usize` index.
    #[inline(always)]
    pub const fn new(index: usize) -> Self {
        Self {
            index,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the underlying `usize` index.
    #[inline(always)]
    pub const fn get(&self) -> usize {
        self.index
    }

    /// Checks if the index is zero.
    #[inline(always)]
    pub const fn is_zero(&self) -> bool {
        self.index == 0
    }

    /// Maps this `TypedIndex<T>` to a `TypedIndex<U>` with the same underlying index.
    #[inline(always)]
    pub fn map<U>(&self) -> TypedIndex<U>
    where
        U: TypedIndexTag,
    {
        TypedIndex::new(self.index)
    }
}

impl<T> std::fmt::Debug for TypedIndex<T>
where
    T: TypedIndexTag,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", T::NAME, self.index)
    }
}

impl<T> std::fmt::Display for TypedIndex<T>
where
    T: TypedIndexTag,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", T::NAME, self.index)
    }
}

impl<T> From<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

impl<T> From<TypedIndex<T>> for usize
where
    T: TypedIndexTag,
{
    fn from(typed_index: TypedIndex<T>) -> Self {
        typed_index.index
    }
}

impl<T> std::ops::Add<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: usize) -> Self::Output {
        Self::new(self.index + rhs)
    }
}

impl<T> std::ops::Sub<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: usize) -> Self::Output {
        Self::new(self.index - rhs)
    }
}

impl<T> std::ops::Mul<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: usize) -> Self::Output {
        Self::new(self.index * rhs)
    }
}

impl<T> std::ops::Div<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: usize) -> Self::Output {
        Self::new(self.index / rhs)
    }
}

impl<T> std::ops::Rem<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: usize) -> Self::Output {
        Self::new(self.index % rhs)
    }
}

impl<T> std::ops::AddAssign<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: usize) {
        self.index += rhs;
    }
}

impl<T> std::ops::SubAssign<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: usize) {
        self.index -= rhs;
    }
}

impl<T> std::ops::MulAssign<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: usize) {
        self.index *= rhs;
    }
}

impl<T> std::ops::DivAssign<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: usize) {
        self.index /= rhs;
    }
}

impl<T> std::ops::RemAssign<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    #[inline(always)]
    fn rem_assign(&mut self, rhs: usize) {
        self.index %= rhs;
    }
}

impl<T> PartialEq<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    #[inline(always)]
    fn eq(&self, other: &usize) -> bool {
        self.index == *other
    }
}

impl<T> PartialEq<TypedIndex<T>> for usize
where
    T: TypedIndexTag,
{
    #[inline(always)]
    fn eq(&self, other: &TypedIndex<T>) -> bool {
        *self == other.index
    }
}

impl<T> PartialOrd<usize> for TypedIndex<T>
where
    T: TypedIndexTag,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &usize) -> Option<std::cmp::Ordering> {
        self.index.partial_cmp(other)
    }
}

impl<T> PartialOrd<TypedIndex<T>> for usize
where
    T: TypedIndexTag,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &TypedIndex<T>) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Define a dummy tag for testing purposes
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    struct TestTag;

    impl TypedIndexTag for TestTag {
        const NAME: &'static str = "TestIdx";
    }

    // Type alias for convenience inside tests
    type TestIndex = TypedIndex<TestTag>;

    #[test]
    fn test_new_and_get() {
        let idx = TestIndex::new(10);
        assert_eq!(idx.get(), 10);
    }

    #[test]
    fn test_conversions() {
        // From usize
        let idx: TestIndex = 42.into();
        assert_eq!(idx.get(), 42);

        // Into usize
        let val: usize = idx.into();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_debug_and_display() {
        let idx = TestIndex::new(7);
        // Uses the NAME const from the trait
        assert_eq!(format!("{}", idx), "TestIdx(7)");
        assert_eq!(format!("{:?}", idx), "TestIdx(7)");
    }

    #[test]
    fn test_arithmetic_ops() {
        let idx = TestIndex::new(10);

        // Test operators (consuming self/copy)
        assert_eq!((idx + 5).get(), 15);
        assert_eq!((idx - 5).get(), 5);
        assert_eq!((idx * 2).get(), 20);
        assert_eq!((idx / 2).get(), 5);
        assert_eq!((idx % 3).get(), 1);
    }

    #[test]
    fn test_assignment_ops() {
        let mut idx = TestIndex::new(10);

        idx += 5;
        assert_eq!(idx.get(), 15);

        idx -= 5;
        assert_eq!(idx.get(), 10);

        idx *= 2;
        assert_eq!(idx.get(), 20);

        idx /= 4;
        assert_eq!(idx.get(), 5);

        idx %= 2;
        assert_eq!(idx.get(), 1);
    }

    #[test]
    fn test_is_zero() {
        let zero = TestIndex::new(0);
        let non_zero = TestIndex::new(1);

        assert!(zero.is_zero());
        assert!(!non_zero.is_zero());
    }

    #[test]
    fn test_ordering_and_equality() {
        let a = TestIndex::new(1);
        let b = TestIndex::new(2);
        let c = TestIndex::new(2);

        assert!(a < b);
        assert!(b > a);
        assert_eq!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn test_hash_works_consistently() {
        use std::collections::HashSet;

        let a = TestIndex::new(1);
        let b = TestIndex::new(2);
        let c = TestIndex::new(1);

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);

        // c should be considered the same key as a
        assert!(set.contains(&c));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_arithmetic_does_not_mutate_original() {
        let idx = TestIndex::new(10);

        let plus = idx + 5;
        let minus = idx - 3;
        let mul = idx * 2;
        let div = idx / 5;
        let rem = idx % 3;

        // original unchanged
        assert_eq!(idx.get(), 10);

        assert_eq!(plus.get(), 15);
        assert_eq!(minus.get(), 7);
        assert_eq!(mul.get(), 20);
        assert_eq!(div.get(), 2);
        assert_eq!(rem.get(), 1);
    }

    #[test]
    #[should_panic]
    fn test_divide_by_zero_panics() {
        let idx = TestIndex::new(10);
        let _ = idx / 0;
    }

    #[test]
    #[should_panic]
    fn test_rem_by_zero_panics() {
        let idx = TestIndex::new(10);
        let _ = idx % 0;
    }
}
