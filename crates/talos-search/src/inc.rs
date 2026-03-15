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

use crate::oracle::GlobalOracle;
use crate::spool::SolutionPool;
use rand::{Rng, RngExt};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use talos_model::solution::{Solution, SolutionView};

/// Thread-safe incumbent solution store.
///
/// Uses atomic `i64` values for the best and worst objective in the pool so
/// that callers can do a fast, lock-free check before acquiring the mutex.
/// The inner `SolutionPool` is protected by a `Mutex` and only locked
/// internally — consumers never see the lock.
///
/// `T` must be `Copy + Ord` and convertible to `i64` (via `Into<i64>`)
/// so the objective can be stored atomically.
pub struct IncumbentStore<T>
where
    T: Into<i64> + Copy,
{
    best: AtomicI64,
    worst: AtomicI64,
    len: AtomicUsize,
    capacity: usize,
    pool: Mutex<SolutionPool<T>>,
}

impl<T> IncumbentStore<T>
where
    T: Into<i64> + Copy + Ord + From<i64>,
{
    /// Creates a new `IncumbentStore` backed by a pool of `max_size` solutions.
    ///
    /// Both atomic bounds start at `i64::MAX` (no solutions seen yet).
    ///
    /// # Panics
    ///
    /// Panics if `max_size == 0`.
    pub fn new(max_size: usize) -> Self {
        Self {
            best: AtomicI64::new(i64::MAX),
            worst: AtomicI64::new(i64::MAX),
            len: AtomicUsize::new(0),
            capacity: max_size,
            pool: Mutex::new(SolutionPool::new(max_size)),
        }
    }

    /// Returns the best (lowest) objective value seen so far.
    ///
    /// This is a lock-free `Relaxed` load — the value may be slightly stale
    /// under concurrent updates, but is always monotonically non-increasing.
    #[inline]
    pub fn best_objective(&self) -> i64 {
        self.best.load(Ordering::Relaxed)
    }

    /// Returns the worst (highest) objective value currently in the pool.
    ///
    /// Before any solution is inserted, returns `i64::MAX`.
    #[inline]
    pub fn worst_objective(&self) -> i64 {
        self.worst.load(Ordering::Relaxed)
    }

    /// Attempts to insert `solution` into the pool.
    ///
    /// 1. Reads the atomic `worst` value (lock-free).
    /// 2. If the pool is full and the solution is not better, returns `false`
    ///    without acquiring the lock.
    /// 3. Otherwise, locks the pool and delegates to `SolutionPool::try_push`.
    /// 4. On success, updates the atomic `best` and `worst` values.
    ///
    /// Returns `true` if the solution was inserted.
    pub fn try_push_solution(&self, solution: &Solution<T>) -> bool
    where
        T: Copy + Ord,
    {
        let obj: i64 = solution.objective_value().into();

        // Fast path: if the pool is full and this solution is not better than
        // the atomically-cached worst, skip the lock entirely.
        if self.len.load(Ordering::Relaxed) >= self.capacity
            && obj >= self.worst.load(Ordering::Relaxed)
        {
            return false;
        }

        // Slow path: acquire the lock and try to insert.
        let mut pool = self.pool.lock().unwrap();
        if !pool.try_push_solution(solution) {
            return false;
        }

        // Update atomic caches.
        self.len.store(pool.len(), Ordering::Relaxed);
        self.update_atomics(&pool);
        true
    }

    /// Attempts to insert `solution` into the pool.
    ///
    /// 1. Reads the atomic `worst` value (lock-free).
    /// 2. If the pool is full and the solution is not better, returns `false`
    ///    without acquiring the lock.
    /// 3. Otherwise, locks the pool and delegates to `SolutionPool::try_push`.
    /// 4. On success, updates the atomic `best` and `worst` values.
    ///
    /// Returns `true` if the solution was inserted.
    pub fn try_push_solution_view(&self, solution: SolutionView<'_, T>) -> bool
    where
        T: Copy + Ord,
    {
        let obj: i64 = solution.objective_value().into();

        // Fast path: if the pool is full and this solution is not better than
        // the atomically-cached worst, skip the lock entirely.
        if self.len.load(Ordering::Relaxed) >= self.capacity
            && obj >= self.worst.load(Ordering::Relaxed)
        {
            return false;
        }

        // Slow path: acquire the lock and try to insert.
        let mut pool = self.pool.lock().unwrap();
        if !pool.try_push_solution_view(solution) {
            return false;
        }

        // Update atomic caches.
        self.len.store(pool.len(), Ordering::Relaxed);
        self.update_atomics(&pool);
        true
    }

    /// Returns the number of solutions currently in the pool (lock-free).
    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Returns `true` if the pool is empty (lock-free).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the pool has reached its capacity (lock-free).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Returns the capacity of the pool.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Applies `f` to the best (lowest objective) solution, if any.
    ///
    /// The lock is held for the duration of `f`. The caller decides whether
    /// to clone, extract a field, etc.
    pub fn with_best<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Solution<T>) -> R,
    {
        self.pool.lock().unwrap().best().map(f)
    }

    /// Applies `f` to the worst (highest objective) solution, if any.
    ///
    /// The lock is held for the duration of `f`.
    pub fn with_worst<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Solution<T>) -> R,
    {
        self.pool.lock().unwrap().worst().map(f)
    }

    /// Applies `f` to the solution at the given rank (0 = best), if any.
    ///
    /// The lock is held for the duration of `f`.
    pub fn with_get<F, R>(&self, index: usize, f: F) -> Option<R>
    where
        F: FnOnce(&Solution<T>) -> R,
    {
        self.pool.lock().unwrap().get(index).map(f)
    }

    /// Applies `f` to a uniformly random solution from the pool.
    ///
    /// Returns `None` if the pool is empty. The lock is held for the
    /// duration of `f`.
    pub fn with_random<R2, F>(&self, rng: &mut impl Rng, f: F) -> Option<R2>
    where
        F: FnOnce(&Solution<T>) -> R2,
    {
        let pool = self.pool.lock().unwrap();
        let len = pool.len();
        if len == 0 {
            return None;
        }
        let idx = rng.random_range(0..len);
        pool.get(idx).map(f)
    }

    /// Updates the atomic best/worst values from the current pool state.
    fn update_atomics(&self, pool: &SolutionPool<T>) {
        if let Some(best_sol) = pool.best() {
            let best_obj: i64 = best_sol.objective_value().into();
            self.best.fetch_min(best_obj, Ordering::Relaxed);
        }
        if let Some(worst_sol) = pool.worst() {
            let worst_obj: i64 = worst_sol.objective_value().into();
            self.worst.store(worst_obj, Ordering::Relaxed);
        }
    }
}

impl<T> GlobalOracle<T> for IncumbentStore<T>
where
    T: Into<i64> + Copy + Ord + From<i64>,
{
    fn try_push_solution(&self, solution: &Solution<T>) -> bool {
        self.try_push_solution(solution)
    }

    fn try_push_solution_view(&self, solution: SolutionView<'_, T>) -> bool {
        self.try_push_solution_view(solution)
    }

    fn best_objective(&self) -> Option<T> {
        let best_obj = self.best_objective();
        if best_obj == i64::MAX {
            None
        } else {
            Some(T::from(best_obj))
        }
    }

    fn with_best<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Solution<T>) -> R,
    {
        self.with_best(f)
    }

    fn pool_len(&self) -> usize {
        self.len()
    }

    fn with_ranked<F, R>(&self, rank: usize, f: F) -> Option<R>
    where
        F: FnOnce(&Solution<T>) -> R,
    {
        self.with_get(rank, f)
    }
}
