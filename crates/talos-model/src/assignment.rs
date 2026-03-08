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

use crate::index::BerthIndex;

/// Assignment of a single vessel in a state.
///
/// An `Assignment` bundles the start time and berth chosen for a vessel.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Assignment<T> {
    /// The start time assigned to the vessel.
    pub start_time: T,

    /// The berth assigned to the vessel.
    pub berth: BerthIndex,
}

impl<T> Assignment<T> {
    /// Creates a new `Assignment` with the given start time and berth.
    #[inline(always)]
    pub fn new(start_time: T, berth: BerthIndex) -> Self {
        Self { start_time, berth }
    }
}

impl<T> std::fmt::Display for Assignment<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "start_time={}, berth={}", self.start_time, self.berth)
    }
}
