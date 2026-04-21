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

#[cfg(all(unix, not(target_vendor = "apple")))]
use std::time::Duration;

// x32 Linux uses a non-standard type for tv_nsec in timespec.
// See https://sourceware.org/bugzilla/show_bug.cgi?id=16437
#[cfg(all(unix, target_arch = "x86_64", target_pointer_width = "32"))]
#[allow(non_camel_case_types)]
type tv_nsec_t = i64;
#[cfg(all(unix, not(all(target_arch = "x86_64", target_pointer_width = "32"))))]
#[allow(non_camel_case_types)]
type tv_nsec_t = libc::c_long;

#[cfg(all(unix, not(target_vendor = "apple")))]
pub(crate) fn errno() -> libc::c_int {
    #[cfg(target_os = "linux")]
    unsafe {
        *libc::__errno_location()
    }
    #[cfg(target_os = "android")]
    unsafe {
        *libc::__errno()
    }
}

#[cfg(all(unix, not(target_vendor = "apple")))]
#[inline]
pub(crate) fn timeout_to_timespec(timeout: Duration) -> Option<libc::timespec> {
    // Handle overflows early on
    if timeout.as_secs() > libc::time_t::max_value() as u64 {
        return None;
    }

    let now = timespec_now();
    let mut nsec = now.tv_nsec + timeout.subsec_nanos() as tv_nsec_t;
    let mut sec = now.tv_sec.checked_add(timeout.as_secs() as libc::time_t);
    if nsec >= 1_000_000_000 {
        nsec -= 1_000_000_000;
        sec = sec.and_then(|sec| sec.checked_add(1));
    }

    sec.map(|sec| libc::timespec {
        tv_nsec: nsec,
        tv_sec: sec,
    })
}


#[cfg(all(unix, not(target_vendor = "apple")))]
#[inline]
fn timespec_now() -> libc::timespec {
    let mut now = MaybeUninit::<libc::timespec>::uninit();
    let clock = if cfg!(target_os = "android") {
        // Android doesn't support pthread_condattr_setclock, so we need to
        // specify the timeout in CLOCK_REALTIME.
        libc::CLOCK_REALTIME
    } else {
        libc::CLOCK_MONOTONIC
    };
    let r = unsafe { libc::clock_gettime(clock, now.as_mut_ptr()) };
    debug_assert_eq!(r, 0);
    // SAFETY: We know `libc::clock_gettime` has initialized the value.
    unsafe { now.assume_init() }
}
