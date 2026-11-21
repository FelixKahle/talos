// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use pyo3::prelude::*;

pub mod callback;
pub mod edf;
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

    // Local Search - GLS
    m.add_class::<ls::gls::PyLambdaStrategy>()?;
    m.add_class::<ls::gls::PyTrigger>()?;
    m.add_class::<ls::gls::PyDecay>()?;
    m.add_class::<ls::gls::PyGlsConfig>()?;

    // Earliest-Deadline-First (EDF) heuristic
    m.add_function(wrap_pyfunction!(edf::edf_schedule, m)?)?;

    // Solve
    m.add_class::<ls::solve::PyOperator>()?;
    m.add_class::<ls::solve::PyLocalSearchConfig>()?;
    m.add_class::<ls::solve::PySolver>()?;
    m.add_function(wrap_pyfunction!(ls::solve::heuristic_gls_lambda, m)?)?;

    Ok(())
}
