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

/// Lambda scaling strategy.
#[pyclass(name = "LambdaStrategy", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PyLambdaStrategy {
    Static,
    Dynamic,
    Additive,
}

/// When GLS fires its penalization step.
#[pyclass(name = "Trigger", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PyTrigger {
    OnExhaustion = 0,
    AfterNonImprovements = 1,
    AfterMoves = 2,
}

/// Penalty decay strategy.
#[pyclass(name = "Decay", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PyDecay {
    NoDecay = 0,
    Geometric = 1,
}

#[pyclass(name = "GlsConfig", from_py_object)]
#[derive(Clone)]
pub struct PyGlsConfig {
    #[pyo3(get)]
    pub lambda_strategy: PyLambdaStrategy,
    #[pyo3(get)]
    pub lambda_initial: Option<f64>,
    #[pyo3(get)]
    pub lambda_inc_step: f64,
    #[pyo3(get)]
    pub lambda_dec_step: f64,
    #[pyo3(get)]
    pub lambda_min: Option<f64>,
    #[pyo3(get)]
    pub lambda_max: Option<f64>,
    #[pyo3(get)]
    pub trigger: PyTrigger,
    #[pyo3(get)]
    pub trigger_threshold: u64,
    #[pyo3(get)]
    pub decay: PyDecay,
    #[pyo3(get)]
    pub decay_factor: f64,
    #[pyo3(get)]
    pub decay_period: u64,
    #[pyo3(get)]
    pub reset_on_best: bool,
}

#[pymethods]
impl PyGlsConfig {
    #[new]
    #[pyo3(signature = (
        lambda_strategy = PyLambdaStrategy::Dynamic,
        lambda_initial = None,
        lambda_inc_step = 0.1,
        lambda_dec_step = 0.1,
        lambda_min = None,
        lambda_max = None,
        trigger = PyTrigger::OnExhaustion,
        trigger_threshold = 1000000,
        decay = PyDecay::NoDecay,
        decay_factor = 0.9,
        decay_period = 10,
        reset_on_best = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lambda_strategy: PyLambdaStrategy,
        lambda_initial: Option<f64>,
        lambda_inc_step: f64,
        lambda_dec_step: f64,
        lambda_min: Option<f64>,
        lambda_max: Option<f64>,
        trigger: PyTrigger,
        trigger_threshold: u64,
        decay: PyDecay,
        decay_factor: f64,
        decay_period: u64,
        reset_on_best: bool,
    ) -> Self {
        Self {
            lambda_strategy,
            lambda_initial,
            lambda_inc_step,
            lambda_dec_step,
            lambda_min,
            lambda_max,
            trigger,
            trigger_threshold,
            decay,
            decay_factor,
            decay_period,
            reset_on_best,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GlsConfig(strategy={:?}, trigger={:?}, decay={:?})",
            self.lambda_strategy, self.trigger, self.decay
        )
    }
}
