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

use core::cell::{Cell, UnsafeCell};
use core::mem::MaybeUninit;
use std::time::Instant;

use crate::auxiliary::timeout_to_timespec;

// Helper type for putting a thread to sleep until some other thread wakes it up
pub struct ThreadParker {
    should_park: Cell<bool>,
    mutex: UnsafeCell<libc::pthread_mutex_t>,
    condvar: UnsafeCell<libc::pthread_cond_t>,
    initialized: Cell<bool>,
}

impl ThreadParker {
    #[allow(clippy::new_without_default)]
    #[inline]
    pub fn new() -> ThreadParker {
        ThreadParker {
            should_park: Cell::new(false),
            mutex: UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER),
            condvar: UnsafeCell::new(libc::PTHREAD_COND_INITIALIZER),
            initialized: Cell::new(false),
        }
    }

    // #[inline]
    // pub fn prepare_park(&self) {
    //     self.should_park.set(true);
    //     if !self.initialized.get() {
    //         self.init();
    //         self.initialized.set(true);
    //     }
    // }

    // #[inline]
    // pub fn timed_out(&self) -> bool {
    //     // We need to grab the mutex here because another thread may be
    //     // concurrently executing UnparkHandle::unpark, which is done without
    //     // holding the queue lock.
    //     let r = libc::pthread_mutex_lock(self.mutex.get());
    //     debug_assert_eq!(r, 0);
    //     let should_park = self.should_park.get();
    //     let r = libc::pthread_mutex_unlock(self.mutex.get());
    //     debug_assert_eq!(r, 0);
    //     should_park
    // }

    #[inline]
    pub fn park(&self) {
        let r = unsafe { libc::pthread_mutex_lock(self.mutex.get()) };
        debug_assert_eq!(r, 0);
        while self.should_park.get() {
            let r = unsafe { libc::pthread_cond_wait(self.condvar.get(), self.mutex.get()) };
            debug_assert_eq!(r, 0);
        }
        let r = unsafe { libc::pthread_mutex_unlock(self.mutex.get()) };
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub fn park_until(&self, timeout: Instant) -> bool {
        let r = unsafe { libc::pthread_mutex_lock(self.mutex.get()) };
        debug_assert_eq!(r, 0);
        while self.should_park.get() {
            let now = Instant::now();
            if timeout <= now {
                let r = unsafe { libc::pthread_mutex_unlock(self.mutex.get()) };
                debug_assert_eq!(r, 0);
                return false;
            }

            if let Some(ts) = timeout_to_timespec(timeout - now) {
                let r = unsafe {
                    libc::pthread_cond_timedwait(self.condvar.get(), self.mutex.get(), &ts)
                };
                if ts.tv_sec < 0 {
                    // On some systems, negative timeouts will return EINVAL. In
                    // that case we won't sleep and will just busy loop instead,
                    // which is the best we can do.
                    debug_assert!(r == 0 || r == libc::ETIMEDOUT || r == libc::EINVAL);
                } else {
                    debug_assert!(r == 0 || r == libc::ETIMEDOUT);
                }
            } else {
                // Timeout calculation overflowed, just sleep indefinitely
                let r = unsafe { libc::pthread_cond_wait(self.condvar.get(), self.mutex.get()) };
                debug_assert_eq!(r, 0);
            }
        }
        let r = unsafe { libc::pthread_mutex_unlock(self.mutex.get()) };
        debug_assert_eq!(r, 0);
        true
    }
    #[inline]
    pub fn unpark(&self) {
        let r = unsafe { libc::pthread_mutex_lock(self.mutex.get()) };
        debug_assert_eq!(r, 0);

        self.should_park.set(false);

        // We notify while holding the lock here to avoid races with the target
        // thread. In particular, the thread could exit after we unlock the
        // mutex, which would make the condvar access invalid memory.
        let r = unsafe { libc::pthread_cond_signal(self.condvar.get()) };
        debug_assert_eq!(r, 0);
        let r = unsafe { libc::pthread_mutex_unlock(self.mutex.get()) };
        debug_assert_eq!(r, 0);
    }

    fn init(&self) {
        let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();
        let r = unsafe { libc::pthread_condattr_init(attr.as_mut_ptr()) };
        debug_assert_eq!(r, 0);
        let r =
            unsafe { libc::pthread_condattr_setclock(attr.as_mut_ptr(), libc::CLOCK_MONOTONIC) };
        debug_assert_eq!(r, 0);
        let r = unsafe { libc::pthread_cond_init(self.condvar.get(), attr.as_ptr()) };
        debug_assert_eq!(r, 0);
        let r = unsafe { libc::pthread_condattr_destroy(attr.as_mut_ptr()) };
        debug_assert_eq!(r, 0);
    }
}

impl Drop for ThreadParker {
    #[inline]
    fn drop(&mut self) {
        // On DragonFly pthread_mutex_destroy() returns EINVAL if called on a
        // mutex that was just initialized with libc::PTHREAD_MUTEX_INITIALIZER.
        // Once it is used (locked/unlocked) or pthread_mutex_init() is called,
        // this behaviour no longer occurs. The same applies to condvars.
        unsafe {
            let r = libc::pthread_mutex_destroy(self.mutex.get());
            debug_assert!(r == 0 || r == libc::EINVAL);
            let r = libc::pthread_cond_destroy(self.condvar.get());
            debug_assert!(r == 0 || r == libc::EINVAL);
        }
    }
}
