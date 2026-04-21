/*
 * Copyright (c) 2026 Trung Tran <tqtrungse@gmail.com>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

use core::ffi;
use core::sync::atomic::{AtomicI8, Ordering};
use std::time::Instant;

#[allow(non_camel_case_types)]
type dispatch_semaphore_t = *mut ffi::c_void;

#[allow(non_camel_case_types)]
type dispatch_time_t = u64;

const DISPATCH_TIME_NOW: dispatch_time_t = 0;
const DISPATCH_TIME_FOREVER: dispatch_time_t = !0;

// Contained in libSystem.dylib, which is linked by default.
unsafe extern "C" {
    fn dispatch_time(when: dispatch_time_t, delta: i64) -> dispatch_time_t;
    fn dispatch_semaphore_create(val: isize) -> dispatch_semaphore_t;
    fn dispatch_semaphore_wait(dsema: dispatch_semaphore_t, timeout: dispatch_time_t) -> isize;
    fn dispatch_semaphore_signal(dsema: dispatch_semaphore_t) -> isize;
    fn dispatch_release(object: *mut ffi::c_void);
}

const EMPTY: i8 = 0;
const NOTIFIED: i8 = 1;
const PARKED: i8 = -1;

pub struct ThreadParker {
    semaphore: dispatch_semaphore_t,
    state: AtomicI8,
}

impl ThreadParker {
    #[allow(clippy::new_without_default)]
    #[inline]
    pub fn new() -> Self {
        let semaphore = unsafe { dispatch_semaphore_create(0) };
        assert!(
            !semaphore.is_null(),
            "failed to create dispatch semaphore for thread synchronization"
        );
        Self {
            semaphore,
            state: AtomicI8::new(EMPTY),
        }
    }

    pub fn park(&self) {
        // The semaphore counter must be zero at this point, because unparking
        // threads will not actually increase it until we signalled that we
        // are waiting.

        // Change NOTIFIED to EMPTY and EMPTY to PARKED.
        if self.state.fetch_sub(1, Ordering::Acquire) == NOTIFIED {
            return;
        }

        // Another thread may increase the semaphore counter from this point on.
        // If it is faster than us, we will decrement it again immediately below.
        // If we are faster, we wait.

        // Ensure that the semaphore counter has actually been decremented, even
        // if the call timed out for some reason.
        while unsafe { dispatch_semaphore_wait(self.semaphore, DISPATCH_TIME_FOREVER) } != 0 {}

        // At this point, the semaphore counter is zero again.

        // We were definitely woken up, so we don't need to check the state.
        // Still, we need to reset the state using a swap to observe the state
        // change with acquire ordering.
        self.state.swap(EMPTY, Ordering::Acquire);
    }

    pub fn park_until(&self, timeout: Instant) -> bool {
        if self.state.fetch_sub(1, Ordering::Acquire) == NOTIFIED {
            return true;
        }
        
        loop {
            let now = Instant::now();
            if timeout <= now {
                return false;
            }
            let dur = timeout - now;
            let nanos = dur.as_nanos().try_into().unwrap_or(i64::MAX);
            let timeout = unsafe { dispatch_time(DISPATCH_TIME_NOW, nanos) };
            let is_timeout = unsafe { dispatch_semaphore_wait(self.semaphore, timeout) } != 0;
           
            let state = self.state.swap(EMPTY, Ordering::Acquire);
            if state == NOTIFIED && is_timeout {
                // If the state was NOTIFIED but semaphore_wait returned without
                // decrementing the count because of a timeout, it means another
                // thread is about to call semaphore_signal. We must wait for that
                // to happen to ensure the semaphore count is reset.
                while unsafe { dispatch_semaphore_wait(self.semaphore, DISPATCH_TIME_FOREVER) } != 0 {}
                return true;
            } else {
                // Either a timeout occurred, and we reset the state before any thread
                // tried to wake us up, or we were woken up and reset the state,
                // making sure to observe the state change with acquire ordering.
                // Either way, the semaphore counter is now zero again.
            }
        }

    }

    #[inline]
    pub fn unpark(&self) {
        let state = self.state.swap(NOTIFIED, Ordering::Release);
        if state == PARKED {
            unsafe {
                dispatch_semaphore_signal(self.semaphore);
            }
        }
    }
}

impl Drop for ThreadParker {
    fn drop(&mut self) {
        // SAFETY:
        // We always ensure that the semaphore count is reset, so this will
        // never cause an exception.
        unsafe {
            dispatch_release(self.semaphore);
        }
    }
}
