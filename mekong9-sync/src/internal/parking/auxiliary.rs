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

use core::cell::Cell;
use core::ptr;
use core::sync::atomic;
use core::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use mekong9_parker::ThreadParker;

use crate::internal::parking::UnparkToken;
use crate::internal::parking::time::FairTimeout;
use crate::internal::{ParkResult, SpinWait};

const IDLE: u64 = 0;
const LOCKED_BIT: u64 = 1;
const QUEUE_LOCKED_BIT: u64 = 2;
const TIMEOUT_BIT: u64 = 4;
const QUEUE_MASK: u64 = !7;
const TOKEN_HANDOFF: UnparkToken = UnparkToken(TIMEOUT_BIT as usize);

macro_rules! park {
    ($self: expr, $state: expr) => {
        with_thread_data(|thread_data| {
            // We typically use thread-local storage and reuses it through multiple lock() calls.
            //
            // So we have to reset to original data: reset unpark_token, prev pointer and
            // queue_tail pointer.
            thread_data.unpark_token.set(UnparkToken(0));

            // Add our thread to the front of the queue
            let queue_head = $state.queue_head();
            if queue_head.is_null() {
                thread_data.queue_tail.set(thread_data);
                thread_data.prev.set(ptr::null());
            } else {
                thread_data.queue_tail.set(ptr::null());
                thread_data.prev.set(ptr::null());
                thread_data.next.set(queue_head);
            }
            if let Err(new_state) = $self.state.compare_exchange_weak(
                $state,
                $state.set_queue_head(thread_data),
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                $state = new_state;
                return ParkResult::Invalid;
            }

            // Sleep until we are woken up by an unlock
            // Ignoring unused unsafe, since it's only a few platforms where this is unsafe.
            thread_data.parker.park();
            return ParkResult::Unparked(thread_data.unpark_token.get());
        })
    };
}

macro_rules! link_prev_nodes {
    ($queue_tail: expr, $current: expr) => {
        loop {
            $queue_tail = unsafe { (*$current).queue_tail.get() };
            if !$queue_tail.is_null() {
                break;
            }
            unsafe {
                let next = (*$current).next.get();
                (*next).prev.set($current);
                $current = next;
            }
        }
    };
}

trait LockState {
    fn is_queue_locked(self) -> bool;
    fn queue_head(self) -> *const ThreadData;
    fn set_queue_head(self, thread_data: *const ThreadData) -> Self;
}

impl LockState for u64 {
    #[inline]
    fn is_queue_locked(self) -> bool {
        self & QUEUE_LOCKED_BIT != 0
    }

    #[cfg(target_pointer_width = "32")]
    #[inline]
    fn queue_head(self) -> *const ThreadData {
        ((self & QUEUE_MASK) >> 3) as *const ThreadData
    }

    #[cfg(target_pointer_width = "64")]
    #[inline]
    fn queue_head(self) -> *const ThreadData {
        // On 64 bits, value of pointer is always divisible by 8, so the rightest 3 bits
        // is always zero.
        (self & QUEUE_MASK) as *const ThreadData
    }

    #[cfg(target_pointer_width = "32")]
    #[inline]
    fn set_queue_head(self, thread_data: *const ThreadData) -> Self {
        // On 32 bits, value of pointer is always divisible by 4, so the rightest 2 bits
        // is always zero. But we need 3 bits for state, so we use 64 bits variable  on 32 bits.
        (self & !QUEUE_MASK) | (thread_data as *const _ as u64) << 3
    }

    #[cfg(target_pointer_width = "64")]
    #[inline]
    fn set_queue_head(self, thread_data: *const ThreadData) -> Self {
        // On 64 bits, value of pointer is always divisible by 8, so the rightest 3 bits
        // is always zero.
        (self & !QUEUE_MASK) | thread_data as *const _ as u64
    }
}

struct ThreadData {
    parker: ThreadParker,
    unpark_token: Cell<UnparkToken>,

    // Linked list of threads in the queue. The queue is split into two parts:
    // the processed part and the unprocessed part. When new nodes are added to
    // the list, they only have the next pointer set, and queue_tail is null.
    //
    // Nodes are processed with the queue lock held, which consists of setting
    // the prev pointer for each node and setting the queue_tail pointer on the
    // first processed node of the list.
    //
    // This setup allows nodes to be added to the queue without a lock, while
    // still allowing O(1) removal of nodes from the processed part of the list.
    // The only cost is the O(n) processing, but this only needs to be done
    // once for each node, and therefore isn't too expensive.
    queue_tail: Cell<*const ThreadData>,
    prev: Cell<*const ThreadData>,
    next: Cell<*const ThreadData>,
}

impl ThreadData {
    #[inline]
    fn new() -> ThreadData {
        assert!(align_of::<ThreadData>() >= size_of::<u32>());
        ThreadData {
            parker: ThreadParker::new(),
            unpark_token: Cell::new(UnparkToken(0)),
            queue_tail: Cell::new(ptr::null()),
            prev: Cell::new(ptr::null()),
            next: Cell::new(ptr::null()),
        }
    }
}

pub struct RawMutex {
    state: AtomicU64,
    fair_timeout: FairTimeout,
}

impl RawMutex {
    #[inline]
    pub fn new() -> Self {
        let state = AtomicU64::new(IDLE);
        let seed = &state as *const _ as u32;
        Self {
            state,
            fair_timeout: FairTimeout::new(Instant::now(), seed),
        }
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        self.state
            .compare_exchange_weak(IDLE, LOCKED_BIT, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    #[inline]
    pub fn lock(&self) {
        if self
            .state
            .compare_exchange_weak(IDLE, LOCKED_BIT, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
        self.lock_slow();
    }

    #[inline]
    pub fn unlock(&self) {
        let state = self.state.fetch_sub(LOCKED_BIT, Ordering::Release);
        if state.is_queue_locked() || state.queue_head().is_null() {
            return;
        }
        self.unlock_slow();
    }

    #[inline]
    pub fn unlock_fair(&self) {
        let mut state: u64;
        let mut bit = LOCKED_BIT;
        if self.fair_timeout.is_timeout() {
            state = self.state.fetch_or(TIMEOUT_BIT, Ordering::Release);
            if !state.queue_head().is_null() {
                self.unlock_slow_fair();
                return;
            }
            bit |= TIMEOUT_BIT;
        }

        state = self.state.fetch_and(!bit, Ordering::Release);
        if state.is_queue_locked() || state.queue_head().is_null() {
            return;
        }
        self.unlock_slow();
    }

    #[cold]
    fn lock_slow(&self) {
        let mut spin_wait = SpinWait::new();
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            // Grab the lock if it isn't locked, even if there is a queue on it
            if state & LOCKED_BIT == 0 {
                match self.state.compare_exchange_weak(
                    state,
                    state | LOCKED_BIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(new_state) => state = new_state,
                }
                continue;
            }

            // If there is no queue, try spinning a few times
            if state.queue_head().is_null() && spin_wait.spin() {
                state = self.state.load(Ordering::Relaxed);
                continue;
            }

            // Get our thread data and prepare it for parking
            match park!(self, state) {
                ParkResult::Unparked(TOKEN_HANDOFF) => return,
                ParkResult::Unparked(_) => {
                    // Loop back and try locking again
                    spin_wait.reset();
                    state = self.state.load(Ordering::Relaxed);
                }
                ParkResult::Invalid => (),
            }
        }
    }

    #[cold]
    fn unlock_slow(&self) {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            // We just unlocked the WordLock. Just check if there is a thread
            // to wake up. If the queue is locked then another thread is already
            // taking care of waking up a thread.
            if state.is_queue_locked() || state.queue_head().is_null() {
                return;
            }

            // Try to grab the queue lock
            match self.state.compare_exchange_weak(
                state,
                state | QUEUE_LOCKED_BIT,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_state) => state = new_state,
            }
        }

        // Now we have the queue lock and the queue is non-empty
        'outer: loop {
            // First, we need to fill in the prev pointers for any newly added
            // threads. We do this until we reach a node that we previously
            // processed, which has a non-null queue_tail pointer.
            let queue_head = state.queue_head();
            let mut queue_tail;
            let mut current = queue_head;
            link_prev_nodes!(queue_tail, current);

            // Set queue_tail on the queue head to indicate that the whole list
            // has prev pointers set correctly.
            unsafe {
                (*queue_head).queue_tail.set(queue_tail);
            }

            // If the WordLock is locked, then there is no point waking up a
            // thread now. Instead, we let the next unlocker take care of waking
            // up a thread.
            if state & LOCKED_BIT != 0 {
                match self.state.compare_exchange_weak(
                    state,
                    state & !QUEUE_LOCKED_BIT,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(new_state) => state = new_state,
                }

                // Need an acquire fence before reading the new queue
                fence_acquire(&self.state);
                continue;
            }

            // Remove the last thread from the queue and unlock the queue
            let new_tail = unsafe { (*queue_tail).prev.get() };
            if new_tail.is_null() {
                loop {
                    match self.state.compare_exchange_weak(
                        state,
                        state & (LOCKED_BIT | TIMEOUT_BIT),
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(new_state) => state = new_state,
                    }

                    // If the compare_exchange failed because a new thread was
                    // added to the queue then we need to re-scan the queue to
                    // find the previous element.
                    if state.queue_head().is_null() {
                        continue;
                    } else {
                        // Need an acquire fence before reading the new queue
                        fence_acquire(&self.state);
                        continue 'outer;
                    }
                }
            } else {
                unsafe {
                    (*queue_head).queue_tail.set(new_tail);
                }
                self.state.fetch_and(!QUEUE_LOCKED_BIT, Ordering::Release);
            }

            // Finally, wake up the thread we removed from the queue. Note that
            // we don't need to worry about any races here since the thread is
            // guaranteed to be sleeping right now, and we are the only one who
            // can wake it up.
            unsafe {
                (*queue_tail).parker.unpark();
            }
            break;
        }
    }

    #[cold]
    fn unlock_slow_fair(&self) {
        let mut spin_wait = SpinWait::new();
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if state.is_queue_locked() {
                if !spin_wait.spin() {
                    spin_wait.reset();
                }
                state = self.state.load(Ordering::Relaxed);
                continue;
            }
            match self.state.compare_exchange_weak(
                state,
                state | QUEUE_LOCKED_BIT,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_state) => state = new_state,
            }
        }
        if state.queue_head().is_null() {
            self.state.fetch_and(QUEUE_MASK, Ordering::Release);
            return;
        }

        'outer: loop {
            // First, we need to fill in the prev pointers for any newly added
            // threads. We do this until we reach a node that we previously
            // processed, which has a non-null queue_tail pointer.
            let queue_head = state.queue_head();
            let mut queue_tail;
            let mut current = queue_head;
            link_prev_nodes!(queue_tail, current);

            // Set queue_tail on the queue head to indicate that the whole list
            // has prev pointers set correctly.
            unsafe {
                (*queue_head).queue_tail.set(queue_tail);
            }

            // Remove the last thread from the queue and unlock the queue
            let new_tail = unsafe { (*queue_tail).prev.get() };
            if new_tail.is_null() {
                loop {
                    match self.state.compare_exchange_weak(
                        state,
                        state & LOCKED_BIT,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(new_state) => state = new_state,
                    }

                    // If the compare_exchange failed because a new thread was
                    // added to the queue then we need to re-scan the queue to
                    // find the previous element.
                    if state.queue_head().is_null() {
                        continue;
                    } else {
                        // Need an acquire fence before reading the new queue
                        fence_acquire(&self.state);
                        continue 'outer;
                    }
                }
            } else {
                unsafe {
                    (*queue_head).queue_tail.set(new_tail);
                }
                self.state
                    .fetch_and(!(QUEUE_LOCKED_BIT | TIMEOUT_BIT), Ordering::Release);
            }

            // Finally, wake up the thread we removed from the queue. Note that
            // we don't need to worry about any races here since the thread is
            // guaranteed to be sleeping right now, and we are the only one who
            // can wake it up.
            unsafe {
                (*queue_tail).unpark_token.set(TOKEN_HANDOFF);
                (*queue_tail).parker.unpark();
            }
            return;
        }
    }
}

const QUEUE_CLEAR_BIT: u64 = 1;

pub struct ParkingQueue {
    state: AtomicU64,
}

impl ParkingQueue {
    #[inline]
    pub fn new() -> Self {
        Self {
            state: AtomicU64::new(IDLE),
        }
    }

    pub fn park(&self, retry: impl Fn() -> bool) {
        let mut spin_wait = SpinWait::new();
        loop {
            if retry() {
                return;
            }
            if spin_wait.spin() {
                continue;
            }

            let state = self.state.load(Ordering::Relaxed);
            with_thread_data(|thread_data| {
                // We typically use thread-local storage and reuses it through multiple lock() calls.
                // So we have to reset to original data: reset unpark_token, prev pointer and
                // queue_tail pointer.
                thread_data.unpark_token.set(UnparkToken(0));

                // Add our thread to the front of the queue
                let queue_head = state.queue_head();
                if queue_head.is_null() {
                    thread_data.queue_tail.set(thread_data);
                    thread_data.prev.set(ptr::null());
                } else {
                    thread_data.queue_tail.set(ptr::null());
                    thread_data.prev.set(ptr::null());
                    thread_data.next.set(queue_head);
                }
                if self
                    .state
                    .compare_exchange_weak(
                        state,
                        state.set_queue_head(thread_data),
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_err()
                {
                    spin_wait.reset();
                    return;
                }

                if retry() {
                    self.abort_self(thread_data);
                    return;
                }
                thread_data.parker.park();
            })
        }
    }

    pub fn unpark_all(&self) {
        let mut spin_wait = SpinWait::new();
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if state & QUEUE_CLEAR_BIT != 0 {
                return;
            }
            if state.is_queue_locked() {
                if !spin_wait.spin() {
                    spin_wait.reset();
                }
                state = self.state.load(Ordering::Relaxed);
                continue;
            }

            match self.state.compare_exchange_weak(
                state,
                QUEUE_CLEAR_BIT,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_state) => state = new_state,
            }
        }

        loop {
            let mut queue_head = state.queue_head();
            if queue_head.is_null() {
                self.state.fetch_and(!QUEUE_CLEAR_BIT, Ordering::Release);
                return;
            }

            while !queue_head.is_null() {
                let new_head = unsafe { (*queue_head).next.get() };
                unsafe {
                    (*queue_head).parker.unpark();
                }
                queue_head = new_head;
            }
            state = self.state.load(Ordering::Relaxed);
        }
    }

    fn abort_self(&self, thread_data: *const ThreadData) {
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if state.is_queue_locked()
                || state & QUEUE_CLEAR_BIT != 0
                || state.queue_head().is_null()
            {
                return;
            }

            match self.state.compare_exchange_weak(
                state,
                state | QUEUE_LOCKED_BIT,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_state) => state = new_state,
            }
        }

        'outer: loop {
            let mut curr = state.queue_head();
            let mut prev: *const ThreadData = ptr::null();

            while !curr.is_null() {
                if thread_data != curr {
                    prev = curr;
                    curr = unsafe { (*curr).next.get() };
                    continue;
                }

                // In a head.
                if prev.is_null() {
                    let new_queue_head = unsafe { (*curr).next.get() };
                    if !new_queue_head.is_null() {
                        unsafe {
                            (*new_queue_head)
                                .queue_tail
                                .set((*thread_data).queue_tail.get())
                        };
                    }

                    match self.state.compare_exchange_weak(
                        state,
                        state.set_queue_head(new_queue_head) & !QUEUE_LOCKED_BIT,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => (),
                        Err(new_state) => {
                            state = new_state;
                            // Need an acquire fence before reading the new queue
                            fence_acquire(&self.state);
                            continue 'outer;
                        }
                    }
                } else {
                    unsafe { (*prev).next.set((*curr).next.get()) };
                    self.state.fetch_and(!QUEUE_LOCKED_BIT, Ordering::Release);
                }

                unsafe {
                    (*thread_data).parker.unpark();
                }
                return;
            }
            self.state.fetch_and(!QUEUE_LOCKED_BIT, Ordering::Release);
        }
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "windows",
    all(target_vendor = "apple", not(miri))
))]
#[inline(always)]
fn with_thread_data<T>(f: impl FnOnce(&ThreadData) -> T) -> T {
    let thread_data = ThreadData::new();
    f(&thread_data)
}

#[cfg(any(
    all(
        unix,
        not(any(
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd",
            target_os = "openbsd",
            all(target_vendor = "apple", not(miri))
        ))
    ),
    target_os = "teeos"
))]
#[inline]
fn with_thread_data<T>(f: impl FnOnce(&ThreadData) -> T) -> T {
    let mut thread_data_ptr = ptr::null();
    thread_local!(static THREAD_DATA: ThreadData = ThreadData::new());
    if let Ok(tls_thread_data) = THREAD_DATA.try_with(|x| x as *const ThreadData) {
        thread_data_ptr = tls_thread_data;
    }

    f(unsafe { &*thread_data_ptr })
}

/// Thread-Sanitizer only has partial fence support, so when running under it, we
/// try and avoid false positives by using a discarded acquire load instead.
#[inline]
pub(crate) fn fence_acquire(atomic: &AtomicU64) {
    if cfg!(any(feature = "tsan_enabled", sanitizer = "thread")) {
        // TSan needs to see a load operation to understand what variable 'fence' is attached to.
        let _ = atomic.load(Ordering::Relaxed);
    }
    atomic::fence(Ordering::Acquire);
}
