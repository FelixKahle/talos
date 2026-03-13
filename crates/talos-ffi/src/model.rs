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

use std::slice;
use talos_core::math::interval::ClosedOpenInterval;
use talos_model::index::{BerthIndex, VesselIndex};
use talos_model::model::Model;
use talos_model::model::ProcessingTime;

/// A C-compatible representation of a closed-open interval with `i64` bounds.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfiClosedOpenIntervalI64 {
    pub start_inclusive: i64,
    pub end_exclusive: i64,
}

/// Creates a new `Model<i64>` from C-compatible raw pointers.
///
/// # Safety
///
/// * `arrival_times_ptr`, `latest_departure_times_ptr`, and `vessel_weights_ptr` must be valid, non-null pointers to arrays of exactly `num_vessels` elements.
/// * `processing_times_ptr` must be a valid, non-null pointer to an array of exactly `num_vessels * num_berths` elements.
/// * `opening_intervals_ptrs` and `opening_intervals_lens` must be valid, non-null pointers to arrays of exactly `num_berths` elements.
/// * Each pointer inside the array pointed to by `opening_intervals_ptrs` must be a valid, non-null pointer to an array of `FfiClosedOpenIntervalI64` structs of the length specified by the corresponding element in `opening_intervals_lens`.
/// * The memory referenced by these pointers must not be mutated concurrently while this function executes.
/// * The returned pointer transfers ownership of the allocated memory to the caller. It must eventually be freed using `talos_model_free` to prevent memory leaks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_new(
    num_vessels: usize,
    num_berths: usize,
    arrival_times_ptr: *const i64,
    latest_departure_times_ptr: *const i64,
    vessel_weights_ptr: *const i64,
    processing_times_ptr: *const i64, // Raw values: < 0 is treated as None
    opening_intervals_ptrs: *const *const FfiClosedOpenIntervalI64,
    opening_intervals_lens: *const usize,
) -> *mut Model<i64> {
    assert!(
        !arrival_times_ptr.is_null(),
        "arrival_times_ptr must not be null"
    );
    assert!(
        !latest_departure_times_ptr.is_null(),
        "latest_departure_times_ptr must not be null"
    );
    assert!(
        !vessel_weights_ptr.is_null(),
        "vessel_weights_ptr must not be null"
    );
    assert!(
        !processing_times_ptr.is_null(),
        "processing_times_ptr must not be null"
    );
    assert!(
        !opening_intervals_ptrs.is_null(),
        "opening_intervals_ptrs must not be null"
    );
    assert!(
        !opening_intervals_lens.is_null(),
        "opening_intervals_lens must not be null"
    );

    let arrival_times = unsafe { slice::from_raw_parts(arrival_times_ptr, num_vessels).to_vec() };
    let latest_departure_times =
        unsafe { slice::from_raw_parts(latest_departure_times_ptr, num_vessels).to_vec() };
    let vessel_weights = unsafe { slice::from_raw_parts(vessel_weights_ptr, num_vessels).to_vec() };

    let raw_processing_times =
        unsafe { slice::from_raw_parts(processing_times_ptr, num_vessels * num_berths) };
    let processing_times: Vec<ProcessingTime<i64>> = raw_processing_times
        .iter()
        .map(|&val| ProcessingTime::from_raw(val))
        .collect();

    let mut opening_times = Vec::with_capacity(num_berths);
    let intervals_ptrs_slice = unsafe { slice::from_raw_parts(opening_intervals_ptrs, num_berths) };
    let intervals_lens_slice = unsafe { slice::from_raw_parts(opening_intervals_lens, num_berths) };

    for i in 0..num_berths {
        let ptr = intervals_ptrs_slice[i];
        let len = intervals_lens_slice[i];

        assert!(
            !(ptr.is_null() && len > 0),
            "opening_intervals_ptrs[{i}] must not be null when len > 0"
        );

        let c_intervals = if len > 0 {
            unsafe { slice::from_raw_parts(ptr, len) }
        } else {
            &[]
        };

        let rust_intervals = c_intervals
            .iter()
            .map(|ffi_iv| ClosedOpenInterval::new(ffi_iv.start_inclusive, ffi_iv.end_exclusive))
            .collect::<Vec<_>>();

        opening_times.push(rust_intervals);
    }

    let model = Model::new(
        num_vessels,
        num_berths,
        arrival_times,
        latest_departure_times,
        vessel_weights,
        processing_times,
        opening_times,
    );

    Box::into_raw(Box::new(model))
}

/// Overrides the state of an existing `Model<i64>` with new data, reusing
/// internal allocations to maximize performance.
///
/// Returns `true` if successful, or `false` if any constraints failed
/// (e.g., null pointers or invalid jagged array states).
///
/// # Safety
///
/// * `model_ptr` must be a valid, mutable pointer to a `Model<i64>` created by `talos_model_new`.
/// * `arrival_times_ptr`, `latest_departure_times_ptr`, and `vessel_weights_ptr` must be valid, non-null pointers to arrays of exactly `num_vessels` elements.
/// * `processing_times_ptr` must be a valid, non-null pointer to an array of exactly `num_vessels * num_berths` elements.
/// * `opening_intervals_ptrs` and `opening_intervals_lens` must be valid, non-null pointers to arrays of exactly `num_berths` elements.
/// * Each pointer inside the array pointed to by `opening_intervals_ptrs` must be a valid, non-null pointer to an array of `FfiClosedOpenIntervalI64` structs of the length specified by the corresponding element in `opening_intervals_lens`.
/// * The memory referenced by these data arrays must not be mutated concurrently during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_override(
    model_ptr: *mut Model<i64>,
    num_vessels: usize,
    num_berths: usize,
    arrival_times_ptr: *const i64,
    latest_departure_times_ptr: *const i64,
    vessel_weights_ptr: *const i64,
    processing_times_ptr: *const i64,
    opening_intervals_ptrs: *const *const FfiClosedOpenIntervalI64,
    opening_intervals_lens: *const usize,
) -> bool {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    assert!(
        !arrival_times_ptr.is_null(),
        "arrival_times_ptr must not be null"
    );
    assert!(
        !latest_departure_times_ptr.is_null(),
        "latest_departure_times_ptr must not be null"
    );
    assert!(
        !vessel_weights_ptr.is_null(),
        "vessel_weights_ptr must not be null"
    );
    assert!(
        !processing_times_ptr.is_null(),
        "processing_times_ptr must not be null"
    );
    assert!(
        !opening_intervals_ptrs.is_null(),
        "opening_intervals_ptrs must not be null"
    );
    assert!(
        !opening_intervals_lens.is_null(),
        "opening_intervals_lens must not be null"
    );

    let model = unsafe { &mut *model_ptr };

    assert_eq!(num_vessels, model.num_vessels(), "num_vessels mismatch");
    assert_eq!(num_berths, model.num_berths(), "num_berths mismatch");

    let arrival_times = unsafe { slice::from_raw_parts(arrival_times_ptr, num_vessels) };
    let latest_departure_times =
        unsafe { slice::from_raw_parts(latest_departure_times_ptr, num_vessels) };
    let vessel_weights = unsafe { slice::from_raw_parts(vessel_weights_ptr, num_vessels) };

    let raw_processing_times =
        unsafe { slice::from_raw_parts(processing_times_ptr, num_vessels * num_berths) };
    let processing_times: Vec<ProcessingTime<i64>> = raw_processing_times
        .iter()
        .map(|&val| ProcessingTime::from_raw(val))
        .collect();

    let intervals_ptrs_slice = unsafe { slice::from_raw_parts(opening_intervals_ptrs, num_berths) };
    let intervals_lens_slice = unsafe { slice::from_raw_parts(opening_intervals_lens, num_berths) };

    let mut opening_times_temp = Vec::with_capacity(num_berths);
    for i in 0..num_berths {
        let ptr = intervals_ptrs_slice[i];
        let len = intervals_lens_slice[i];

        assert!(
            !(ptr.is_null() && len > 0),
            "opening_intervals_ptrs[{i}] must not be null when len > 0"
        );

        let c_intervals = if len > 0 {
            unsafe { slice::from_raw_parts(ptr, len) }
        } else {
            &[]
        };

        let rust_intervals: Vec<_> = c_intervals
            .iter()
            .map(|ffi_iv| ClosedOpenInterval::new(ffi_iv.start_inclusive, ffi_iv.end_exclusive))
            .collect();

        opening_times_temp.push(rust_intervals);
    }

    let opening_times_slices: Vec<&[ClosedOpenInterval<i64>]> =
        opening_times_temp.iter().map(|v| v.as_slice()).collect();

    model.override_from(
        num_vessels,
        num_berths,
        arrival_times,
        latest_departure_times,
        vessel_weights,
        processing_times.as_slice(),
        &opening_times_slices,
    );

    true
}

// -----------------------------------------------------------------------------
// Accessors / Getters
// -----------------------------------------------------------------------------

/// Returns the number of vessels in the model.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_num_vessels(model_ptr: *const Model<i64>) -> usize {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    unsafe { (*model_ptr).num_vessels() }
}

/// Returns the number of berths in the model.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_num_berths(model_ptr: *const Model<i64>) -> usize {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    unsafe { (*model_ptr).num_berths() }
}

/// Returns a raw pointer to the start of the `arrival_times` array.
/// The length of this array is exactly `num_vessels`.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
/// * The returned pointer is strictly bound to the lifetime of the `Model`. The caller must not dereference this pointer after `talos_model_free` is called.
/// * The returned memory must be treated as strictly read-only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_arrival_times_ptr(model_ptr: *const Model<i64>) -> *const i64 {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    unsafe { (*model_ptr).vessel_arrival_times().as_ptr() }
}

/// Returns a raw pointer to the start of the `latest_departure_times` array.
/// The length of this array is exactly `num_vessels`.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
/// * The returned pointer is strictly bound to the lifetime of the `Model`. The caller must not dereference this pointer after `talos_model_free` is called.
/// * The returned memory must be treated as strictly read-only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_latest_departure_times_ptr(
    model_ptr: *const Model<i64>,
) -> *const i64 {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    unsafe { (*model_ptr).vessel_latest_departure_times().as_ptr() }
}

/// Returns a raw pointer to the start of the `vessel_weights` array.
/// The length of this array is exactly `num_vessels`.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
/// * The returned pointer is strictly bound to the lifetime of the `Model`. The caller must not dereference this pointer after `talos_model_free` is called.
/// * The returned memory must be treated as strictly read-only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_vessel_weights_ptr(
    model_ptr: *const Model<i64>,
) -> *const i64 {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    unsafe { (*model_ptr).vessel_weights().as_ptr() }
}

/// Returns a raw pointer to the flattened `processing_times` array.
/// The length of this array is exactly `num_vessels * num_berths`.
/// Because `ProcessingTime<i64>` uses a transparent sentinel encoding,
/// C can read this directly as an array of `i64` where negative values mean `None`.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
/// * The returned pointer is strictly bound to the lifetime of the `Model`. The caller must not dereference this pointer after `talos_model_free` is called.
/// * The returned memory must be treated as strictly read-only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_processing_times_ptr(
    model_ptr: *const Model<i64>,
) -> *const i64 {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    // Cast is safe due to #[repr(transparent)] on ProcessingTime
    unsafe { (*model_ptr).vessel_processing_times_matrix().as_ptr() as *const i64 }
}

/// Returns the arrival time for a single vessel.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_vessel_arrival_time(
    model_ptr: *const Model<i64>,
    vessel_index: usize,
) -> i64 {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    let model = unsafe { &*model_ptr };
    assert!(
        vessel_index < model.num_vessels(),
        "vessel_index out of bounds"
    );
    model.vessel_arrival_time(VesselIndex::new(vessel_index))
}

/// Returns the latest departure time for a single vessel.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_vessel_latest_departure_time(
    model_ptr: *const Model<i64>,
    vessel_index: usize,
) -> i64 {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    let model = unsafe { &*model_ptr };
    assert!(
        vessel_index < model.num_vessels(),
        "vessel_index out of bounds"
    );
    model.vessel_latest_departure_time(VesselIndex::new(vessel_index))
}

/// Returns the weight for a single vessel.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_vessel_weight(
    model_ptr: *const Model<i64>,
    vessel_index: usize,
) -> i64 {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    let model = unsafe { &*model_ptr };
    assert!(
        vessel_index < model.num_vessels(),
        "vessel_index out of bounds"
    );
    model.vessel_weight(VesselIndex::new(vessel_index))
}

/// Returns the processing time for a (vessel, berth) pair.
///
/// Returns the raw `i64` value. Negative values indicate that the assignment
/// is forbidden (equivalent to `None`).
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_vessel_processing_time(
    model_ptr: *const Model<i64>,
    vessel_index: usize,
    berth_index: usize,
) -> i64 {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    let model = unsafe { &*model_ptr };
    assert!(
        vessel_index < model.num_vessels(),
        "vessel_index out of bounds"
    );
    assert!(
        berth_index < model.num_berths(),
        "berth_index out of bounds"
    );
    model
        .vessel_processing_time(VesselIndex::new(vessel_index), BerthIndex::new(berth_index))
        .raw()
}

/// Returns `true` if the vessel is allowed to dock at the specified berth.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_vessel_allowed_on_berth(
    model_ptr: *const Model<i64>,
    vessel_index: usize,
    berth_index: usize,
) -> bool {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    let model = unsafe { &*model_ptr };
    assert!(
        vessel_index < model.num_vessels(),
        "vessel_index out of bounds"
    );
    assert!(
        berth_index < model.num_berths(),
        "berth_index out of bounds"
    );
    model.vessel_allowed_on_berth(VesselIndex::new(vessel_index), BerthIndex::new(berth_index))
}

/// Returns the number of opening-time intervals for a given berth.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_berth_opening_times_len(
    model_ptr: *const Model<i64>,
    berth_index: usize,
) -> usize {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    let model = unsafe { &*model_ptr };
    assert!(
        berth_index < model.num_berths(),
        "berth_index out of bounds"
    );
    model
        .berth_opening_times(BerthIndex::new(berth_index))
        .len()
}

/// Copies the opening-time intervals for a given berth into a caller-provided buffer.
///
/// The caller must allocate `out_buf` with at least `buf_len` elements. Use
/// `talos_model_berth_opening_times_len` to query the required length first.
///
/// Returns the number of intervals actually written (capped at `buf_len`).
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer to a `Model<i64>`.
/// * `out_buf` must be a valid pointer to an array of at least `buf_len` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_berth_opening_times(
    model_ptr: *const Model<i64>,
    berth_index: usize,
    out_buf: *mut FfiClosedOpenIntervalI64,
    buf_len: usize,
) -> usize {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    assert!(!out_buf.is_null(), "out_buf must not be null");
    let model = unsafe { &*model_ptr };
    assert!(
        berth_index < model.num_berths(),
        "berth_index out of bounds"
    );

    let intervals = model.berth_opening_times(BerthIndex::new(berth_index));
    let count = intervals.len().min(buf_len);
    let out_slice = unsafe { slice::from_raw_parts_mut(out_buf, count) };

    for (i, iv) in intervals.iter().take(count).enumerate() {
        out_slice[i] = FfiClosedOpenIntervalI64 {
            start_inclusive: iv.start(),
            end_exclusive: iv.end(),
        };
    }

    count
}

/// Frees a `Model<i64>` previously allocated by `talos_model_new`.
///
/// # Safety
///
/// * `model_ptr` must be a valid pointer returned by `talos_model_new`.
/// * The pointer must not be used after this function returns (no use-after-free).
/// * The pointer must not be freed more than once (no double-free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn talos_model_free(model_ptr: *mut Model<i64>) {
    assert!(!model_ptr.is_null(), "model_ptr must not be null");
    drop(unsafe { Box::from_raw(model_ptr) });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_talos_model_ffi_lifecycle_and_accessors() {
        let num_vessels = 2;
        let num_berths = 2;

        let arrivals = [0, 10];
        let departures = [100, 110];
        let weights = [1, 2];
        let processing = [5, -1, 7, 8]; // -1 represents None in the FFI mapping

        let b0_intervals = [FfiClosedOpenIntervalI64 {
            start_inclusive: 0,
            end_exclusive: 50,
        }];
        let b1_intervals = [FfiClosedOpenIntervalI64 {
            start_inclusive: 0,
            end_exclusive: 60,
        }];

        let intervals_ptrs = [b0_intervals.as_ptr(), b1_intervals.as_ptr()];
        let intervals_lens = [b0_intervals.len(), b1_intervals.len()];

        // 1. Create the model
        let model_ptr = unsafe {
            talos_model_new(
                num_vessels,
                num_berths,
                arrivals.as_ptr(),
                departures.as_ptr(),
                weights.as_ptr(),
                processing.as_ptr(),
                intervals_ptrs.as_ptr(),
                intervals_lens.as_ptr(),
            )
        };

        assert!(!model_ptr.is_null());

        // 2. Validate Accessors
        assert_eq!(unsafe { talos_model_num_vessels(model_ptr) }, 2);
        assert_eq!(unsafe { talos_model_num_berths(model_ptr) }, 2);

        // Check arrival times read via ptr
        let arr_ptr = unsafe { talos_model_arrival_times_ptr(model_ptr) };
        assert!(!arr_ptr.is_null());
        let read_arrivals = unsafe { slice::from_raw_parts(arr_ptr, 2) };
        assert_eq!(read_arrivals, &[0, 10]);

        // Check processing times read via ptr
        let proc_ptr = unsafe { talos_model_processing_times_ptr(model_ptr) };
        assert!(!proc_ptr.is_null());
        let read_processing = unsafe { slice::from_raw_parts(proc_ptr, 4) };
        assert_eq!(read_processing, &[5, -1, 7, 8]);

        // Per-vessel scalar accessors
        assert_eq!(unsafe { talos_model_vessel_arrival_time(model_ptr, 0) }, 0);
        assert_eq!(unsafe { talos_model_vessel_arrival_time(model_ptr, 1) }, 10);
        assert_eq!(
            unsafe { talos_model_vessel_latest_departure_time(model_ptr, 0) },
            100
        );
        assert_eq!(
            unsafe { talos_model_vessel_latest_departure_time(model_ptr, 1) },
            110
        );
        assert_eq!(unsafe { talos_model_vessel_weight(model_ptr, 0) }, 1);
        assert_eq!(unsafe { talos_model_vessel_weight(model_ptr, 1) }, 2);

        // Per-(vessel, berth) processing time
        assert_eq!(
            unsafe { talos_model_vessel_processing_time(model_ptr, 0, 0) },
            5
        );
        assert_eq!(
            unsafe { talos_model_vessel_processing_time(model_ptr, 0, 1) },
            -1
        ); // None sentinel
        assert_eq!(
            unsafe { talos_model_vessel_processing_time(model_ptr, 1, 0) },
            7
        );
        assert_eq!(
            unsafe { talos_model_vessel_processing_time(model_ptr, 1, 1) },
            8
        );

        // vessel_allowed_on_berth
        assert!(unsafe { talos_model_vessel_allowed_on_berth(model_ptr, 0, 0) });
        assert!(!unsafe { talos_model_vessel_allowed_on_berth(model_ptr, 0, 1) }); // -1 => not allowed
        assert!(unsafe { talos_model_vessel_allowed_on_berth(model_ptr, 1, 0) });
        assert!(unsafe { talos_model_vessel_allowed_on_berth(model_ptr, 1, 1) });

        // Berth opening times
        assert_eq!(
            unsafe { talos_model_berth_opening_times_len(model_ptr, 0) },
            1
        );
        assert_eq!(
            unsafe { talos_model_berth_opening_times_len(model_ptr, 1) },
            1
        );

        let mut buf = [FfiClosedOpenIntervalI64 {
            start_inclusive: 0,
            end_exclusive: 0,
        }; 4];
        let written =
            unsafe { talos_model_berth_opening_times(model_ptr, 0, buf.as_mut_ptr(), buf.len()) };
        assert_eq!(written, 1);
        assert_eq!(buf[0].start_inclusive, 0);
        assert_eq!(buf[0].end_exclusive, 50);

        let written =
            unsafe { talos_model_berth_opening_times(model_ptr, 1, buf.as_mut_ptr(), buf.len()) };
        assert_eq!(written, 1);
        assert_eq!(buf[0].start_inclusive, 0);
        assert_eq!(buf[0].end_exclusive, 60);

        // 3. Free the model
        unsafe {
            talos_model_free(model_ptr);
        }
    }

    #[test]
    fn test_talos_model_ffi_override() {
        let num_vessels = 2;
        let num_berths = 2;

        let arrivals = [0, 10];
        let departures = [100, 110];
        let weights = [1, 2];
        let processing = [5, -1, 7, 8];

        let b0_intervals = [FfiClosedOpenIntervalI64 {
            start_inclusive: 0,
            end_exclusive: 50,
        }];
        let b1_intervals = [FfiClosedOpenIntervalI64 {
            start_inclusive: 0,
            end_exclusive: 60,
        }];

        let intervals_ptrs = [b0_intervals.as_ptr(), b1_intervals.as_ptr()];
        let intervals_lens = [b0_intervals.len(), b1_intervals.len()];

        let model_ptr = unsafe {
            talos_model_new(
                num_vessels,
                num_berths,
                arrivals.as_ptr(),
                departures.as_ptr(),
                weights.as_ptr(),
                processing.as_ptr(),
                intervals_ptrs.as_ptr(),
                intervals_lens.as_ptr(),
            )
        };

        // Prepare new data to override with
        let new_arrivals = [5, 15];
        let new_departures = [100, 110];
        let new_weights = [1, 2];
        let new_processing = [-1, 10, 12, -1];

        let new_b0_intervals = [FfiClosedOpenIntervalI64 {
            start_inclusive: 10,
            end_exclusive: 20,
        }];
        let new_b1_intervals = [FfiClosedOpenIntervalI64 {
            start_inclusive: 30,
            end_exclusive: 40,
        }];
        let new_intervals_ptrs = [new_b0_intervals.as_ptr(), new_b1_intervals.as_ptr()];
        let new_intervals_lens = [new_b0_intervals.len(), new_b1_intervals.len()];

        let success = unsafe {
            talos_model_override(
                model_ptr,
                num_vessels,
                num_berths,
                new_arrivals.as_ptr(),
                new_departures.as_ptr(),
                new_weights.as_ptr(),
                new_processing.as_ptr(),
                new_intervals_ptrs.as_ptr(),
                new_intervals_lens.as_ptr(),
            )
        };
        assert!(success);

        // Verify the pointers read updated data
        let arr_ptr = unsafe { talos_model_arrival_times_ptr(model_ptr) };
        let read_arrivals = unsafe { slice::from_raw_parts(arr_ptr, 2) };
        assert_eq!(read_arrivals, &[5, 15]);

        let proc_ptr = unsafe { talos_model_processing_times_ptr(model_ptr) };
        let read_processing = unsafe { slice::from_raw_parts(proc_ptr, 4) };
        assert_eq!(read_processing, &[-1, 10, 12, -1]);

        unsafe {
            talos_model_free(model_ptr);
        }
    }
}
