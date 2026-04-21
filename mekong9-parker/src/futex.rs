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
use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
#[cfg(target_os = "windows")]
use {
    core::ffi::c_void,
    std::sync::OnceLock,
    windows_sys::Win32::Foundation::{ERROR_TIMEOUT, GetLastError},
    windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA},
    windows_sys::Win32::System::Threading::{INFINITE, WakeByAddressAll, WakeByAddressSingle},
};

use mekong9_utils::unlikely;

#[cfg(all(unix, not(target_vendor = "apple")))]
use crate::auxiliary::{errno, timeout_to_timespec};

const PARKED: u32 = u32::MAX;
const EMPTY: u32 = 0;
const NOTIFIED: u32 = 1;

/// The platform thread parker implementation.
pub struct ThreadParker {
    state: AtomicU32,
}

impl ThreadParker {
    #[allow(clippy::new_without_default)]
    #[inline]
    pub fn new() -> ThreadParker {
        ThreadParker {
            state: AtomicU32::new(EMPTY),
        }
    }

    // /// Prepares the parker. This should be called before adding it to the queue.
    // #[inline]
    // pub fn prepare_park(&self) {
    //     self.state.fetch_or(PARKED, Ordering::Relaxed);
    // }

    // /// Checks if the park timed out. This should be called while holding the
    // /// queue lock after `park_until` has returned false.
    // #[inline]
    // pub fn timed_out(&self) -> bool {
    //     self.state.load(Ordering::Relaxed) != 0
    // }

    /// Parks the thread until it is unparked. This should be called after it has
    /// been added to the queue, after unlocking the queue.
    #[inline]
    pub fn park(&self) {
        if self.state.fetch_sub(1, Ordering::Acquire) == NOTIFIED {
            return;
        }
        loop {
            wait(&self.state, PARKED);
            // Change NOTIFIED=>EMPTY and return in that case.
            if self
                .state
                .compare_exchange(NOTIFIED, EMPTY, Ordering::Acquire, Ordering::Acquire)
                .is_ok()
            {
                return;
            } else {
                // Spurious wake up. We loop to try again.
            }
        }
    }

    /// Parks the thread until it is unparked or the timeout is reached. This
    /// should be called after it has been added to the queue, after unlocking
    /// the queue.
    ///
    /// Returns TRUE if we were unparked, FALSE if we timed out.
    #[inline]
    pub fn park_until(&self, timeout: Instant) -> bool {
        // Change NOTIFIED=>EMPTY or EMPTY=>PARKED, and directly return in the
        // first case.
        if self.state.fetch_sub(1, Ordering::Acquire) == NOTIFIED {
            return true;
        }

        loop {
            let now = Instant::now();
            if timeout <= now {
                match self.state.swap(EMPTY, Ordering::Acquire) {
                    NOTIFIED => return true, // We got a timeout, but luckily another thread just unparked it.
                    PARKED => return false, // Actual timeout, safely returned state to EMPTY.
                    _ => unreachable!("Inconsistent thread parker state"),
                }
            }
            wait_until(&self.state, PARKED, timeout - now);
            // Change NOTIFIED=>EMPTY and return in that case.
            if self
                .state
                .compare_exchange(NOTIFIED, EMPTY, Ordering::Acquire, Ordering::Acquire)
                .is_ok()
            {
                return true;
            } else {
                // Spurious wake up. We loop to try again.
            }
        }
    }

    /// Handle for a thread that is about to be unparked. We need to mark the thread
    /// as unparked while holding the queue lock, but we delay the actual unparking
    /// until after the queue lock is released.
    ///
    /// Wakes up the parked thread. This should be called after the queue lock is
    /// released to avoid blocking the queue for too long.
    #[inline]
    pub fn unpark(&self) {
        // Change PARKED=>NOTIFIED, EMPTY=>NOTIFIED, or NOTIFIED=>NOTIFIED, and
        // wake the thread in the first case.
        //
        // Note that even NOTIFIED=>NOTIFIED results in a White. This is on
        // purpose, to make sure every unpark() has a release-acquire ordering
        // with park().
        if self.state.swap(NOTIFIED, Ordering::Release) == PARKED {
            wake_one(&self.state);
        }
    }
}

//================
// Linux & android
//================

#[cfg(any(target_os = "linux", target_os = "android"))]
#[inline]
fn wait_until(atom: &AtomicU32, expected: u32, duration: Duration) -> bool {
    let abs_timespec_ptr = timeout_to_timespec(duration)
        .as_ref()
        .map(|ts_ref| ts_ref as *const _)
        .unwrap_or(null());

    let code = unsafe {
        libc::syscall(
            libc::SYS_futex,
            atom as *const AtomicU32,
            libc::FUTEX_WAIT_BITSET | libc::FUTEX_PRIVATE_FLAG | libc::FUTEX_CLOCK_REALTIME,
            expected,
            abs_timespec_ptr as *const libc::c_void,
            0,
            libc::FUTEX_BITSET_MATCH_ANY,
        )
    };

    debug_assert!(code == 0 || code == -1);
    if unlikely(code == -1) {
        debug_assert!(
            errno() == libc::EINTR
                || errno() == libc::EAGAIN
                || (!abs_timespec_ptr.is_null() && errno() == libc::ETIMEDOUT)
        );
        return !abs_timespec_ptr.is_null() && errno() == libc::ETIMEDOUT;
    }
    false
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[inline]
fn wait(atom: &AtomicU32, expected: u32) {
    // https://man7.org/linux/man-pages/man2/futex.2.html
    let code = unsafe {
        libc::syscall(
            libc::SYS_futex,
            atom as *const AtomicU32,
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            null(),
        )
    };

    debug_assert!(code == 0 || code == -1);
    if unlikely(code == -1) {
        debug_assert!(errno() == libc::EINTR || errno() == libc::EAGAIN);
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[inline]
fn wake_one(ptr: *const AtomicU32) {
    let code = unsafe {
        libc::syscall(
            libc::SYS_futex,
            ptr,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            1i32,
        )
    };

    debug_assert!(code == 0 || code == 1 || code == -1);
    if unlikely(code == -1) {
        debug_assert_eq!(errno(), libc::EFAULT);
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[inline]
fn wake_all(ptr: *const AtomicU32) {
    let code = unsafe {
        libc::syscall(
            libc::SYS_futex,
            ptr,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            i32::MAX,
        )
    };

    debug_assert!(code == 0 || code == 1 || code == -1);
    if unlikely(code == -1) {
        debug_assert_eq!(errno(), libc::EFAULT);
    }
}

//=========
// Free BSD
//=========

#[cfg(target_os = "freebsd")]
#[inline]
fn wait_until(atom: &AtomicU32, expected: u32, duration: Duration) -> bool {
    let abs_timespec_ptr = timeout_to_timespec(duration)
        .as_ref()
        .map(|ts_ref| ts_ref as *const _)
        .unwrap_or(null());

    let code = unsafe {
        libc::_umtx_op(
            atom as *const AtomicU32 as *mut _,
            libc::UMTX_OP_WAIT_UINT_PRIVATE | libc::UMTX_OP_WAIT_UINT_PRIVATE_TIMEOUT,
            expected as libc::c_ulong,
            null_mut(),
            abs_timespec_ptr as *mut _,
        )
    };

    debug_assert!(code == 0 || code == -1);
    if unlikely(code == -1) {
        debug_assert!(
            errno() == libc::EINTR
                || errno() == libc::EAGAIN
                || (!abs_timespec_ptr.is_null() && errno() == libc::ETIMEDOUT)
        );
        return !abs_timespec_ptr.is_null() && errno() == libc::ETIMEDOUT;
    }
    false
}

#[cfg(target_os = "freebsd")]
#[inline]
fn wait(atom: &AtomicU32, expected: u32) {
    let code = unsafe {
        libc::_umtx_op(
            atom as *const AtomicU32 as *mut _,
            libc::UMTX_OP_WAIT_UINT_PRIVATE,
            expected as libc::c_ulong,
            null_mut(),
            null_mut(),
        )
    };

    debug_assert!(code == 0 || code == -1);
    if unlikely(code == -1) {
        debug_assert!(errno() == libc::EINTR || errno() == libc::EAGAIN);
    }
}

#[cfg(target_os = "freebsd")]
#[inline]
fn wake_one(ptr: *const AtomicU32) {
    let code = unsafe {
        libc::_umtx_op(
            ptr as *mut _,
            libc::UMTX_OP_WAKE_PRIVATE,
            1,
            null_mut(),
            null_mut(),
        )
    };

    debug_assert!(code == 0 || code == 1 || code == -1);
    if unlikely(code == -1) {
        debug_assert_eq!(errno(), libc::EFAULT);
    }
}

#[cfg(target_os = "freebsd")]
#[inline]
fn wake_all(ptr: *const AtomicU32) {
    let code = unsafe {
        libc::_umtx_op(
            ptr as *mut _,
            libc::UMTX_OP_WAKE_PRIVATE,
            i32::MAX as libc::c_ulong,
            null_mut(),
            null_mut(),
        )
    };

    debug_assert!(code == 0 || code == 1 || code == -1);
    if unlikely(code == -1) {
        debug_assert_eq!(errno(), libc::EFAULT);
    }
}

//=========
// Open BSD
//=========

#[cfg(target_os = "openbsd")]
fn wait_until(atom: &AtomicU32, expected: u32, duration: Duration) -> bool {
    let abs_timespec_ptr = timeout_to_timespec(duration)
        .as_ref()
        .map(|ts_ref| ts_ref as *const _)
        .unwrap_or(null());

    let code = unsafe {
        libc::futex(
            atom as *const AtomicU32 as *mut u32,
            libc::FUTEX_WAIT,
            expected as i32,
            abs_timespec_ptr as *const libc::timespec,
            null_mut(),
        )
    };

    code == 0 || errno() != libc::ETIMEDOUT
}

#[cfg(target_os = "openbsd")]
fn wait(atom: &AtomicU32, expected: u32) {
    unsafe {
        libc::futex(
            atom as *const AtomicU32 as *mut u32,
            libc::FUTEX_WAIT,
            expected as i32,
            null(),
            null_mut(),
        )
    };
}

#[cfg(target_os = "openbsd")]
fn wake_one(ptr: *const AtomicU32) {
    let code = unsafe { libc::futex(ptr as *mut u32, libc::FUTEX_WAKE, 1, null(), null_mut()) };
    if unlikely(code <= 0) {
        debug_assert_eq!(errno(), libc::EFAULT);
    }
}

#[cfg(target_os = "openbsd")]
fn wake_all(ptr: *const AtomicU32) {
    unsafe {
        libc::futex(
            ptr as *mut u32,
            libc::FUTEX_WAKE,
            i32::MAX,
            null(),
            null_mut(),
        );
    };
}

//========
// Windows
//========

#[cfg(target_os = "windows")]
static WAIT_ON_ADDRESS_PTR: OnceLock<Option<WaitOnAddressFn>> = OnceLock::new();

#[cfg(target_os = "windows")]
fn get_cached_ptr() -> Option<WaitOnAddressFn> {
    *WAIT_ON_ADDRESS_PTR.get_or_init(get_wait_on_address_ptr)
}

#[cfg(target_os = "windows")]
type WaitOnAddressFn = unsafe extern "system" fn(
    Address: *const c_void,
    CompareAddress: *const c_void,
    AddressSize: usize,
    dwMilliseconds: u32,
) -> i32;

#[cfg(target_os = "windows")]
fn get_wait_on_address_ptr() -> Option<WaitOnAddressFn> {
    unsafe {
        let mut h_module = LoadLibraryA(b"api-ms-win-core-synch-l1-2-0.dll\0".as_ptr());
        if h_module.is_null() {
            h_module = LoadLibraryA(b"kernelbase.dll\0".as_ptr());
        }
        if h_module.is_null() {
            return None;
        }
        let addr = GetProcAddress(h_module, b"WaitOnAddress\0".as_ptr());
        addr.map(|ptr| std::mem::transmute::<_, WaitOnAddressFn>(ptr))
    }
}

#[cfg(target_os = "windows")]
#[inline]
fn wait_until(atom: &AtomicU32, expected: u32, duration: Duration) -> bool {
    let atom_ptr: *const AtomicU32 = atom;
    let expected_ptr: *const u32 = &expected;
    let timeout = duration
        .as_secs()
        .checked_mul(1000)
        .and_then(|x| x.checked_add((duration.subsec_nanos() as u64).div_ceil(1000000)))
        .map(|ms| {
            if ms > u32::MAX as u64 {
                INFINITE
            } else {
                ms as u32
            }
        })
        .unwrap_or(INFINITE);

    // https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitonaddress
    match get_cached_ptr() {
        Some(wait_on_address_fn) => {
            let success = unsafe {
                wait_on_address_fn(
                    atom_ptr.cast(),
                    expected_ptr.cast(),
                    size_of::<u32>(),
                    timeout,
                )
            };
            if unlikely(success == false.into()) {
                let err = unsafe { GetLastError() };
                debug_assert_eq!(err, ERROR_TIMEOUT);
                return err == ERROR_TIMEOUT;
            }
            false
        }
        None => panic!("only support from Windows 8"),
    }
}

#[cfg(target_os = "windows")]
#[inline]
fn wait(atom: &AtomicU32, expected: u32) {
    wait_until(atom, expected, Duration::MAX);
}

#[cfg(target_os = "windows")]
#[inline]
fn wake_one(atom_ptr: *const AtomicU32) {
    unsafe { WakeByAddressSingle(atom_ptr.cast()) };
}

#[cfg(target_os = "windows")]
#[inline]
fn wake_all(atom_ptr: *const AtomicU32) {
    unsafe { WakeByAddressAll(atom_ptr.cast()) };
}
