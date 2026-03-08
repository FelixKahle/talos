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

use talos_model::index::BerthIndex;

// ----------------------------------------------------------------
// TouchedIndicesIter
// ----------------------------------------------------------------

pub struct TouchedIndicesIter<'a> {
    iter: std::iter::Enumerate<std::slice::Iter<'a, bool>>,
}

impl<'a> Iterator for TouchedIndicesIter<'a> {
    type Item = BerthIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast-forward to the next `true` value and return its index
        for (idx, &is_touched) in &mut self.iter {
            if is_touched {
                return Some(BerthIndex::new(idx));
            }
        }
        None
    }
}

// ----------------------------------------------------------------
// UntouchedBerthsIter
// ----------------------------------------------------------------

pub struct UntouchedBerthsIter<'a> {
    iter: std::iter::Enumerate<std::slice::Iter<'a, bool>>,
}

impl<'a> Iterator for UntouchedBerthsIter<'a> {
    type Item = BerthIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast-forward to the next `false` value and return its index
        for (idx, &is_touched) in &mut self.iter {
            if !is_touched {
                return Some(BerthIndex::new(idx));
            }
        }
        None
    }
}

// ----------------------------------------------------------------
// TouchedBerths
// ----------------------------------------------------------------

/// Tracks which berths have been touched (modified) during a mutation.
///
/// This allows the engine to efficiently identify which berths need to be
/// re-decoded after a mutation, without having to check every berth.
#[repr(transparent)]
#[derive(Clone, PartialEq, Eq)]
pub struct TouchedBerths {
    // Note
    // We might use a bitset in the future
    // but for now a Vec<bool> is simple and efficient enough for our needs.
    touched: Vec<bool>,
}

impl TouchedBerths {
    /// Creates a new touched berths tracker with the given number of berths.
    #[inline(always)]
    pub fn new(num_berths: usize) -> Self {
        Self {
            touched: vec![false; num_berths],
        }
    }

    /// Resizes the tracker to accommodate a new number of berths.
    #[inline(always)]
    pub fn resize(&mut self, num_berths: usize) {
        self.touched.resize(num_berths, false);
    }

    /// Marks a berth as touched.
    #[inline(always)]
    pub fn touch(&mut self, berth: BerthIndex) {
        self.touched[berth.get()] = true;
    }

    /// Marks a berth as touched without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth.get()` is < `num_berths`.
    #[inline(always)]
    pub unsafe fn touch_unchecked(&mut self, berth: BerthIndex) {
        debug_assert!(
            berth.get() < self.touched.len(),
            "called `TouchedBerths::touch_unchecked` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.touched.len()
        );

        unsafe {
            *self.touched.get_unchecked_mut(berth.get()) = true;
        }
    }

    /// Checks if a berth has been touched.
    #[inline(always)]
    pub fn is_touched(&self, berth: BerthIndex) -> bool {
        self.touched[berth.get()]
    }

    /// Checks if a berth has been touched without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth.get()` is < `num_berths`.
    #[inline(always)]
    pub unsafe fn is_touched_unchecked(&self, berth: BerthIndex) -> bool {
        debug_assert!(
            berth.get() < self.touched.len(),
            "called `TouchedBerths::is_touched_unchecked` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.touched.len()
        );

        unsafe { *self.touched.get_unchecked(berth.get()) }
    }

    /// Resets all berths to untouched.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.touched.fill(false);
    }

    /// Returns the number of berths tracked.
    #[inline(always)]
    pub fn num_berths(&self) -> usize {
        self.touched.len()
    }

    /// Returns an iterator over the touched status of each berth.
    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, bool> {
        self.touched.iter()
    }

    /// Returns an iterator over the indices of touched berths.
    #[inline(always)]
    pub fn iter_touched_berths(&self) -> TouchedIndicesIter<'_> {
        TouchedIndicesIter {
            iter: self.touched.iter().enumerate(),
        }
    }

    /// Returns an iterator over the indices of untouched berths.
    #[inline(always)]
    pub fn iter_untouched_berths(&self) -> UntouchedBerthsIter<'_> {
        UntouchedBerthsIter {
            iter: self.touched.iter().enumerate(),
        }
    }

    /// Returns a slice of the touched status of each berth.
    #[inline(always)]
    pub fn as_slice(&self) -> &[bool] {
        &self.touched
    }
}

impl<'a> IntoIterator for &'a TouchedBerths {
    type Item = &'a bool;
    type IntoIter = std::slice::Iter<'a, bool>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.touched.iter()
    }
}

impl<'a> IntoIterator for &'a mut TouchedBerths {
    type Item = &'a mut bool;
    type IntoIter = std::slice::IterMut<'a, bool>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.touched.iter_mut()
    }
}

impl IntoIterator for TouchedBerths {
    type Item = bool;
    type IntoIter = std::vec::IntoIter<bool>;

    fn into_iter(self) -> Self::IntoIter {
        self.touched.into_iter()
    }
}

impl std::fmt::Debug for TouchedBerths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TouchedBerths")
            .field("touched", &self.touched)
            .finish()
    }
}

impl std::fmt::Display for TouchedBerths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Touched Berths: [")?;

        let mut first = true;
        for (idx, &is_touched) in self.touched.iter().enumerate() {
            if is_touched {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{}", idx)?;
                first = false;
            }
        }

        write!(f, "]")
    }
}
