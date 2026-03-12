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

use std::ptr;
use std::slice;
use talos_model::index::BerthIndex;
use talos_model::solution::{Solution, SolutionView};

// ----------------------------------------------------------------
// Owned Solution (heap-allocated, caller owns the lifetime)
// ----------------------------------------------------------------

/// Creates a new owned `Solution<i64>` on the heap.
///
/// The caller receives a pointer that must eventually be freed with
/// `talos_solution_free`. The data behind `berths_ptr` and `start_times_ptr`
/// is **copied** — the caller may free those arrays immediately after this
/// call returns.
///
/// Returns a null pointer if any input pointer is null.
///
/// # Safety
///
/// * `berths_ptr` must be either null or a valid pointer to an array of
///   exactly `num_vessels` `usize` values (berth indices).
/// * `start_times_ptr` must be either null or a valid pointer to an array of
///   exactly `num_vessels` `i64` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_new(
    num_vessels: usize,
    berths_ptr: *const usize,
    start_times_ptr: *const i64,
    objective_value: i64,
) -> *mut Solution<i64> {
    if berths_ptr.is_null() || start_times_ptr.is_null() {
        return ptr::null_mut();
    }

    let raw_berths = unsafe { slice::from_raw_parts(berths_ptr, num_vessels) };
    let start_times = unsafe { slice::from_raw_parts(start_times_ptr, num_vessels) };

    let berths: Vec<BerthIndex> = raw_berths.iter().map(|&b| BerthIndex::new(b)).collect();

    let solution = Solution::new(berths, start_times.to_vec(), objective_value);
    Box::into_raw(Box::new(solution))
}

/// Frees an owned `Solution<i64>` previously allocated by `talos_solution_new`.
///
/// # Safety
///
/// * `solution_ptr` must be either null or a valid pointer returned by
///   `talos_solution_new`.
/// * The pointer must not be used after this call (no use-after-free).
/// * The pointer must not be freed more than once (no double-free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_free(solution_ptr: *mut Solution<i64>) {
    if !solution_ptr.is_null() {
        drop(unsafe { Box::from_raw(solution_ptr) });
    }
}

// ----------------------------------------------------------------
// Owned Solution — Accessors
// ----------------------------------------------------------------

/// Returns the number of vessels.  Returns 0 if `solution_ptr` is null.
///
/// # Safety
///
/// * `solution_ptr` must be either null or a valid pointer to a `Solution<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_num_vessels(solution_ptr: *const Solution<i64>) -> usize {
    if solution_ptr.is_null() {
        return 0;
    }
    unsafe { (*solution_ptr).num_vessels() }
}

/// Returns the objective value.  Returns `i64::MIN` if `solution_ptr` is null.
///
/// # Safety
///
/// * `solution_ptr` must be either null or a valid pointer to a `Solution<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_objective_value(solution_ptr: *const Solution<i64>) -> i64 {
    if solution_ptr.is_null() {
        return i64::MIN;
    }
    unsafe { (*solution_ptr).objective_value() }
}

/// Sets the objective value on an owned solution.
///
/// Does nothing if `solution_ptr` is null.
///
/// # Safety
///
/// * `solution_ptr` must be either null or a valid, mutable pointer to a
///   `Solution<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_set_objective_value(
    solution_ptr: *mut Solution<i64>,
    value: i64,
) {
    if solution_ptr.is_null() {
        return;
    }
    unsafe { (*solution_ptr).set_objective_value(value) }
}

/// Returns a raw pointer to the berths array (length = `num_vessels`).
///
/// Because `BerthIndex` is `#[repr(transparent)]` over `usize`, the returned
/// pointer can be read as `*const usize` on the C side.
///
/// # Safety
///
/// * `solution_ptr` must be either null or a valid pointer to a `Solution<i64>`.
/// * The returned pointer is bound to the lifetime of the `Solution`.
/// * The returned memory must be treated as read-only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_berths_ptr(
    solution_ptr: *const Solution<i64>,
) -> *const usize {
    if solution_ptr.is_null() {
        return ptr::null();
    }
    // Safe cast: BerthIndex is #[repr(transparent)] over usize
    unsafe { (*solution_ptr).berths().as_ptr() as *const usize }
}

/// Returns a raw pointer to the start-times array (length = `num_vessels`).
///
/// # Safety
///
/// * `solution_ptr` must be either null or a valid pointer to a `Solution<i64>`.
/// * The returned pointer is bound to the lifetime of the `Solution`.
/// * The returned memory must be treated as read-only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_start_times_ptr(
    solution_ptr: *const Solution<i64>,
) -> *const i64 {
    if solution_ptr.is_null() {
        return ptr::null();
    }
    unsafe { (*solution_ptr).start_times().as_ptr() }
}

/// Returns the berth index for a single vessel.
///
/// Returns `usize::MAX` if `solution_ptr` is null or `vessel_index` is out of
/// bounds.
///
/// # Safety
///
/// * `solution_ptr` must be either null or a valid pointer to a `Solution<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_berth_for_vessel(
    solution_ptr: *const Solution<i64>,
    vessel_index: usize,
) -> usize {
    if solution_ptr.is_null() {
        return usize::MAX;
    }
    let sol = unsafe { &*solution_ptr };
    if vessel_index >= sol.num_vessels() {
        return usize::MAX;
    }
    sol.berth_for_vessel(talos_model::index::VesselIndex::new(vessel_index))
        .get()
}

/// Returns the start time for a single vessel.
///
/// Returns `i64::MIN` if `solution_ptr` is null or `vessel_index` is out of
/// bounds.
///
/// # Safety
///
/// * `solution_ptr` must be either null or a valid pointer to a `Solution<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_start_time_for_vessel(
    solution_ptr: *const Solution<i64>,
    vessel_index: usize,
) -> i64 {
    if solution_ptr.is_null() {
        return i64::MIN;
    }
    let sol = unsafe { &*solution_ptr };
    if vessel_index >= sol.num_vessels() {
        return i64::MIN;
    }
    *sol.start_time_for_vessel(talos_model::index::VesselIndex::new(vessel_index))
}

// ----------------------------------------------------------------
// Solution View (non-owning, borrows existing data)
// ----------------------------------------------------------------

/// Creates a non-owning `SolutionView` from caller-provided arrays.
///
/// The returned handle **borrows** the memory behind `berths_ptr` and
/// `start_times_ptr` — the caller must keep those arrays alive and unmodified
/// until `talos_solution_view_free` is called.
///
/// Returns null if any input pointer is null.
///
/// # Safety
///
/// * `berths_ptr` must be either null or a valid pointer to an array of
///   exactly `num_vessels` `usize` values that remains valid for the lifetime
///   of the returned handle.
/// * `start_times_ptr` must be either null or a valid pointer to an array of
///   exactly `num_vessels` `i64` values that remains valid for the lifetime
///   of the returned handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_view_new(
    num_vessels: usize,
    berths_ptr: *const usize,
    start_times_ptr: *const i64,
    objective_value: i64,
) -> *mut SolutionView<'static, i64> {
    if berths_ptr.is_null() || start_times_ptr.is_null() {
        return ptr::null_mut();
    }

    // Safe cast: BerthIndex is #[repr(transparent)] over usize
    let berths = unsafe { slice::from_raw_parts(berths_ptr as *const BerthIndex, num_vessels) };
    let start_times = unsafe { slice::from_raw_parts(start_times_ptr, num_vessels) };

    let view = SolutionView::new(berths, start_times, objective_value);
    Box::into_raw(Box::new(view))
}

/// Frees a `SolutionView` handle previously returned by
/// `talos_solution_view_new`.
///
/// # Safety
///
/// * `view_ptr` must be either null or a valid pointer returned by
///   `talos_solution_view_new`.
/// * The pointer must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_view_free(view_ptr: *mut SolutionView<'_, i64>) {
    if !view_ptr.is_null() {
        drop(unsafe { Box::from_raw(view_ptr) });
    }
}

// ----------------------------------------------------------------
// Solution View — Accessors
// ----------------------------------------------------------------

/// Returns the number of vessels.  Returns 0 if `view_ptr` is null.
///
/// # Safety
///
/// * `view_ptr` must be either null or a valid pointer to a `SolutionView`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_view_num_vessels(
    view_ptr: *const SolutionView<'_, i64>,
) -> usize {
    if view_ptr.is_null() {
        return 0;
    }
    unsafe { (*view_ptr).num_vessels() }
}

/// Returns the objective value.  Returns `i64::MIN` if `view_ptr` is null.
///
/// # Safety
///
/// * `view_ptr` must be either null or a valid pointer to a `SolutionView`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_view_objective_value(
    view_ptr: *const SolutionView<'_, i64>,
) -> i64 {
    if view_ptr.is_null() {
        return i64::MIN;
    }
    unsafe { (*view_ptr).objective_value() }
}

/// Returns a raw pointer to the berths array (length = `num_vessels`).
///
/// # Safety
///
/// * `view_ptr` must be either null or a valid pointer to a `SolutionView`.
/// * The returned pointer points into the caller's original array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_view_berths_ptr(
    view_ptr: *const SolutionView<'_, i64>,
) -> *const usize {
    if view_ptr.is_null() {
        return ptr::null();
    }
    unsafe { (*view_ptr).berths().as_ptr() as *const usize }
}

/// Returns a raw pointer to the start-times array (length = `num_vessels`).
///
/// # Safety
///
/// * `view_ptr` must be either null or a valid pointer to a `SolutionView`.
/// * The returned pointer points into the caller's original array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_view_start_times_ptr(
    view_ptr: *const SolutionView<'_, i64>,
) -> *const i64 {
    if view_ptr.is_null() {
        return ptr::null();
    }
    unsafe { (*view_ptr).start_times().as_ptr() }
}

/// Returns the berth index for a single vessel.
///
/// Returns `usize::MAX` if `view_ptr` is null or `vessel_index` is out of
/// bounds.
///
/// # Safety
///
/// * `view_ptr` must be either null or a valid pointer to a `SolutionView`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_view_berth_for_vessel(
    view_ptr: *const SolutionView<'_, i64>,
    vessel_index: usize,
) -> usize {
    if view_ptr.is_null() {
        return usize::MAX;
    }
    let view = unsafe { &*view_ptr };
    if vessel_index >= view.num_vessels() {
        return usize::MAX;
    }
    view.berth_for_vessel(talos_model::index::VesselIndex::new(vessel_index))
        .get()
}

/// Returns the start time for a single vessel.
///
/// Returns `i64::MIN` if `view_ptr` is null or `vessel_index` is out of
/// bounds.
///
/// # Safety
///
/// * `view_ptr` must be either null or a valid pointer to a `SolutionView`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_view_start_time_for_vessel(
    view_ptr: *const SolutionView<'_, i64>,
    vessel_index: usize,
) -> i64 {
    if view_ptr.is_null() {
        return i64::MIN;
    }
    let view = unsafe { &*view_ptr };
    if vessel_index >= view.num_vessels() {
        return i64::MIN;
    }
    *view.start_time_for_vessel(talos_model::index::VesselIndex::new(vessel_index))
}

/// Creates an owned `Solution<i64>` by deep-copying a `SolutionView`.
///
/// The returned pointer must be freed with `talos_solution_free`.
/// Returns null if `view_ptr` is null.
///
/// # Safety
///
/// * `view_ptr` must be either null or a valid pointer to a `SolutionView`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_solution_view_to_owned(
    view_ptr: *const SolutionView<'_, i64>,
) -> *mut Solution<i64> {
    if view_ptr.is_null() {
        return ptr::null_mut();
    }
    let owned = unsafe { (*view_ptr).to_owned_solution() };
    Box::into_raw(Box::new(owned))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    // ---- Owned Solution tests ----

    #[test]
    fn test_solution_lifecycle_and_accessors() {
        let berths: [usize; 3] = [0, 2, 1];
        let starts: [i64; 3] = [10, 20, 30];
        let objective = 42_i64;

        let sol = unsafe { talos_solution_new(3, berths.as_ptr(), starts.as_ptr(), objective) };
        assert!(!sol.is_null());

        assert_eq!(unsafe { talos_solution_num_vessels(sol) }, 3);
        assert_eq!(unsafe { talos_solution_objective_value(sol) }, 42);

        // Bulk pointers
        let b_ptr = unsafe { talos_solution_berths_ptr(sol) };
        assert!(!b_ptr.is_null());
        let read_berths = unsafe { slice::from_raw_parts(b_ptr, 3) };
        assert_eq!(read_berths, &[0, 2, 1]);

        let st_ptr = unsafe { talos_solution_start_times_ptr(sol) };
        assert!(!st_ptr.is_null());
        let read_starts = unsafe { slice::from_raw_parts(st_ptr, 3) };
        assert_eq!(read_starts, &[10, 20, 30]);

        // Per-vessel accessors
        assert_eq!(unsafe { talos_solution_berth_for_vessel(sol, 0) }, 0);
        assert_eq!(unsafe { talos_solution_berth_for_vessel(sol, 1) }, 2);
        assert_eq!(unsafe { talos_solution_berth_for_vessel(sol, 2) }, 1);
        assert_eq!(unsafe { talos_solution_start_time_for_vessel(sol, 0) }, 10);
        assert_eq!(unsafe { talos_solution_start_time_for_vessel(sol, 1) }, 20);
        assert_eq!(unsafe { talos_solution_start_time_for_vessel(sol, 2) }, 30);

        // Out-of-bounds
        assert_eq!(
            unsafe { talos_solution_berth_for_vessel(sol, 99) },
            usize::MAX
        );
        assert_eq!(
            unsafe { talos_solution_start_time_for_vessel(sol, 99) },
            i64::MIN
        );

        // Mutate objective
        unsafe { talos_solution_set_objective_value(sol, 999) };
        assert_eq!(unsafe { talos_solution_objective_value(sol) }, 999);

        unsafe { talos_solution_free(sol) };
    }

    #[test]
    fn test_solution_null_pointers() {
        assert_eq!(unsafe { talos_solution_num_vessels(ptr::null()) }, 0);
        assert_eq!(
            unsafe { talos_solution_objective_value(ptr::null()) },
            i64::MIN
        );
        assert!(unsafe { talos_solution_berths_ptr(ptr::null()) }.is_null());
        assert!(unsafe { talos_solution_start_times_ptr(ptr::null()) }.is_null());
        assert_eq!(
            unsafe { talos_solution_berth_for_vessel(ptr::null(), 0) },
            usize::MAX
        );
        assert_eq!(
            unsafe { talos_solution_start_time_for_vessel(ptr::null(), 0) },
            i64::MIN
        );
        unsafe { talos_solution_set_objective_value(ptr::null_mut(), 0) }; // no-op
        unsafe { talos_solution_free(ptr::null_mut()) }; // no-op

        assert!(unsafe { talos_solution_new(2, ptr::null(), [0i64; 2].as_ptr(), 0) }.is_null());
        assert!(unsafe { talos_solution_new(2, [0usize; 2].as_ptr(), ptr::null(), 0) }.is_null());
    }

    // ---- Solution View tests ----

    #[test]
    fn test_solution_view_lifecycle_and_accessors() {
        let berths: [usize; 2] = [1, 0];
        let starts: [i64; 2] = [100, 200];
        let objective = 77_i64;

        let view =
            unsafe { talos_solution_view_new(2, berths.as_ptr(), starts.as_ptr(), objective) };
        assert!(!view.is_null());

        assert_eq!(unsafe { talos_solution_view_num_vessels(view) }, 2);
        assert_eq!(unsafe { talos_solution_view_objective_value(view) }, 77);

        // Bulk pointers (should point back into original arrays)
        let b_ptr = unsafe { talos_solution_view_berths_ptr(view) };
        assert_eq!(b_ptr, berths.as_ptr());

        let st_ptr = unsafe { talos_solution_view_start_times_ptr(view) };
        assert_eq!(st_ptr, starts.as_ptr());

        // Per-vessel
        assert_eq!(unsafe { talos_solution_view_berth_for_vessel(view, 0) }, 1);
        assert_eq!(unsafe { talos_solution_view_berth_for_vessel(view, 1) }, 0);
        assert_eq!(
            unsafe { talos_solution_view_start_time_for_vessel(view, 0) },
            100
        );
        assert_eq!(
            unsafe { talos_solution_view_start_time_for_vessel(view, 1) },
            200
        );

        // Out-of-bounds
        assert_eq!(
            unsafe { talos_solution_view_berth_for_vessel(view, 99) },
            usize::MAX
        );
        assert_eq!(
            unsafe { talos_solution_view_start_time_for_vessel(view, 99) },
            i64::MIN
        );

        unsafe { talos_solution_view_free(view) };
    }

    #[test]
    fn test_solution_view_to_owned() {
        let berths: [usize; 2] = [0, 1];
        let starts: [i64; 2] = [5, 15];

        let view = unsafe { talos_solution_view_new(2, berths.as_ptr(), starts.as_ptr(), 50) };
        assert!(!view.is_null());

        let owned = unsafe { talos_solution_view_to_owned(view) };
        assert!(!owned.is_null());

        // Owned copy should have the same data
        assert_eq!(unsafe { talos_solution_num_vessels(owned) }, 2);
        assert_eq!(unsafe { talos_solution_objective_value(owned) }, 50);
        assert_eq!(unsafe { talos_solution_berth_for_vessel(owned, 0) }, 0);
        assert_eq!(unsafe { talos_solution_berth_for_vessel(owned, 1) }, 1);
        assert_eq!(unsafe { talos_solution_start_time_for_vessel(owned, 0) }, 5);
        assert_eq!(
            unsafe { talos_solution_start_time_for_vessel(owned, 1) },
            15
        );

        // Free both independently
        unsafe { talos_solution_view_free(view) };
        unsafe { talos_solution_free(owned) };
    }

    #[test]
    fn test_solution_view_null_pointers() {
        assert_eq!(unsafe { talos_solution_view_num_vessels(ptr::null()) }, 0);
        assert_eq!(
            unsafe { talos_solution_view_objective_value(ptr::null()) },
            i64::MIN
        );
        assert!(unsafe { talos_solution_view_berths_ptr(ptr::null()) }.is_null());
        assert!(unsafe { talos_solution_view_start_times_ptr(ptr::null()) }.is_null());
        assert_eq!(
            unsafe { talos_solution_view_berth_for_vessel(ptr::null(), 0) },
            usize::MAX
        );
        assert_eq!(
            unsafe { talos_solution_view_start_time_for_vessel(ptr::null(), 0) },
            i64::MIN
        );
        assert!(unsafe { talos_solution_view_to_owned(ptr::null()) }.is_null());
        unsafe { talos_solution_view_free(ptr::null_mut()) }; // no-op

        assert!(
            unsafe { talos_solution_view_new(2, ptr::null(), [0i64; 2].as_ptr(), 0) }.is_null()
        );
        assert!(
            unsafe { talos_solution_view_new(2, [0usize; 2].as_ptr(), ptr::null(), 0) }.is_null()
        );
    }
}
