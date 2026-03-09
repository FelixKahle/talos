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

use std::iter::Zip;
use std::slice::Iter;
use talos_model::index::{BerthIndex, VesselIndex};

/// A high-level representation of a link change.
///
/// `None` represents a Sentinel (the boundary of the berth).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleGraphEdge {
    pub from: Option<VesselIndex>,
    pub to: Option<VesselIndex>,
}

/// Iterator over broken or created links in a [`ScheduleGraphDiff`].
pub struct LinkIter<'a> {
    inner: Zip<Iter<'a, VesselIndex>, Iter<'a, VesselIndex>>,
    num_vessels: usize,
}

impl<'a> LinkIter<'a> {
    /// Converts a `VesselIndex` to an `Option<VesselIndex>`, treating indices
    /// greater than or equal to `num_vessels` as `None` (sentinels).
    #[inline(always)]
    fn to_option(&self, index: VesselIndex) -> Option<VesselIndex> {
        if index.get() < self.num_vessels {
            Some(index)
        } else {
            None
        }
    }
}

impl<'a> Iterator for LinkIter<'a> {
    type Item = ScheduleGraphEdge;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(&f, &t)| ScheduleGraphEdge {
            from: self.to_option(f),
            to: self.to_option(t),
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Iterator over created links with their associated berth context.
pub struct LinkContextIter<'a> {
    inner: Zip<Iter<'a, VesselIndex>, Iter<'a, VesselIndex>>,
    num_vessels: usize,
}

impl<'a> LinkContextIter<'a> {
    #[inline(always)]
    fn to_option(&self, index: VesselIndex) -> Option<VesselIndex> {
        if index.get() < self.num_vessels {
            Some(index)
        } else {
            None
        }
    }

    #[inline(always)]
    fn to_berth(&self, index: VesselIndex) -> Option<BerthIndex> {
        if index.get() >= self.num_vessels {
            Some(BerthIndex::new(index.get() - self.num_vessels))
        } else {
            None
        }
    }
}

impl<'a> Iterator for LinkContextIter<'a> {
    type Item = (Option<VesselIndex>, Option<VesselIndex>, Option<BerthIndex>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(&f, &t)| {
            let b = self.to_berth(f).or_else(|| self.to_berth(t));
            (self.to_option(f), self.to_option(t), b)
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Iterator over vessel reallocations between berths.
pub struct ReallocationIter<'a> {
    vessels: Iter<'a, VesselIndex>,
    originals: Iter<'a, BerthIndex>,
    targets: Iter<'a, BerthIndex>,
}

impl<'a> Iterator for ReallocationIter<'a> {
    type Item = (VesselIndex, BerthIndex, BerthIndex);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let v = self.vessels.next()?;
        let o = self.originals.next()?;
        let t = self.targets.next()?;
        Some((*v, *o, *t))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.vessels.size_hint()
    }
}

/// A comprehensive diff structure that captures all changes between two schedule graphs
/// states, including broken and created links as well as vessel reallocations between berths.
#[derive(Debug, Clone)]
pub struct ScheduleGraphDiff {
    broken_from: Vec<VesselIndex>,
    broken_to: Vec<VesselIndex>,
    created_from: Vec<VesselIndex>,
    created_to: Vec<VesselIndex>,
    reallocated_vessels: Vec<VesselIndex>,
    original_berths: Vec<BerthIndex>,
    target_berths: Vec<BerthIndex>,
    num_vessels: usize,
}

impl ScheduleGraphDiff {
    #[inline(always)]
    pub fn new(num_vessels: usize) -> Self {
        Self {
            broken_from: Vec::new(),
            broken_to: Vec::new(),
            created_from: Vec::new(),
            created_to: Vec::new(),
            reallocated_vessels: Vec::new(),
            original_berths: Vec::new(),
            target_berths: Vec::new(),
            num_vessels,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.broken_from.clear();
        self.broken_to.clear();
        self.created_from.clear();
        self.created_to.clear();
        self.reallocated_vessels.clear();
        self.original_berths.clear();
        self.target_berths.clear();
    }

    #[inline(always)]
    pub fn push_link_broken(&mut self, from: VesselIndex, to: VesselIndex) {
        self.broken_from.push(from);
        self.broken_to.push(to);
    }

    #[inline(always)]
    pub fn push_link_created(&mut self, from: VesselIndex, to: VesselIndex) {
        self.created_from.push(from);
        self.created_to.push(to);
    }

    #[inline(always)]
    pub fn push_reallocation(&mut self, v: VesselIndex, from: BerthIndex, to: BerthIndex) {
        self.reallocated_vessels.push(v);
        self.original_berths.push(from);
        self.target_berths.push(to);
    }

    #[inline(always)]
    pub fn link_broken_count(&self) -> usize {
        self.broken_from.len()
    }

    #[inline(always)]
    pub fn link_created_count(&self) -> usize {
        self.created_from.len()
    }

    #[inline(always)]
    pub fn broken_links(&self) -> LinkIter<'_> {
        LinkIter {
            inner: self.broken_from.iter().zip(self.broken_to.iter()),
            num_vessels: self.num_vessels,
        }
    }

    #[inline(always)]
    pub fn created_links(&self) -> LinkIter<'_> {
        LinkIter {
            inner: self.created_from.iter().zip(self.created_to.iter()),
            num_vessels: self.num_vessels,
        }
    }

    #[inline(always)]
    pub fn created_links_with_context(&self) -> LinkContextIter<'_> {
        LinkContextIter {
            inner: self.created_from.iter().zip(self.created_to.iter()),
            num_vessels: self.num_vessels,
        }
    }

    #[inline(always)]
    pub fn reallocations(&self) -> ReallocationIter<'_> {
        ReallocationIter {
            vessels: self.reallocated_vessels.iter(),
            originals: self.original_berths.iter(),
            targets: self.target_berths.iter(),
        }
    }
}
