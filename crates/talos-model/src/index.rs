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

use talos_core::utils::index::{TypedIndex, TypedIndexTag};

/// Marker type for indices into a vessel collection.
///
/// Use the `VesselIndex` type alias instead of referring to this marker
/// directly in most code.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VesselIndexTag;

impl TypedIndexTag for VesselIndexTag {
    const NAME: &'static str = "VesselIndex";
}

/// Strongly typed index for vessels.
///
/// This is a zero-cost wrapper around `usize` that prevents mixing
/// vessel indices with other index domains.
pub type VesselIndex = TypedIndex<VesselIndexTag>;

/// Marker type for indices into a berth collection.
///
/// Use the `BerthIndex` type alias instead of referring to this marker
/// directly in most code.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BerthIndexTag;

impl TypedIndexTag for BerthIndexTag {
    const NAME: &'static str = "BerthIndex";
}

/// Strongly typed index for berths.
///
/// This is a zero-cost wrapper around `usize` that prevents mixing
/// berth indices with other index domains.
pub type BerthIndex = TypedIndex<BerthIndexTag>;
