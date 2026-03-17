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

use pyo3::prelude::*;

pub mod eval;
pub mod ls;
pub mod model;
pub mod solution;

/// Talos: Dynamic Berth Allocation Problem solver
#[pymodule]
fn pytalos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Model & Solution
    m.add_class::<model::PyModel>()?;
    m.add_class::<solution::PySolution>()?;

    // Local Search - Outcome
    m.add_class::<ls::outcome::PyTerminationReason>()?;
    m.add_class::<ls::outcome::PySearchResult>()?;

    // Local Search - Engine
    m.add_class::<ls::engine::PyOperator>()?;
    m.add_class::<ls::engine::PyLocalSearchConfig>()?;

    // Local Search - GLS
    m.add_class::<ls::gls::PyLambdaStrategy>()?;
    m.add_class::<ls::gls::PyTrigger>()?;
    m.add_class::<ls::gls::PyDecay>()?;
    m.add_class::<ls::gls::PyGlsConfig>()?;

    // Solve
    m.add_function(wrap_pyfunction!(ls::solve::solve, m)?)?;

    Ok(())
}
