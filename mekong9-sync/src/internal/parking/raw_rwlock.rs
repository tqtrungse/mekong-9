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

// use core::cell::Cell;
// use core::ptr;
// use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
// use std::time::Instant;
// 
// // use mekong9_utils::CachePadded;
// 
// use crate::internal::parking::raw_mutex::RawMutex;
// use crate::internal::parking::thread::ThreadParker;
// use crate::internal::parking::time::FairTimeout;
// use crate::internal::{ParkResult, ParkToken, SpinWait, UnparkToken};
// 
// /// There is at least one thread in the main queue.
// const PARKED_BIT: usize = 0b0001;
// /// There is a parked thread holding WRITER_BIT. WRITER_BIT must be set.
// const WRITER_PARKED_BIT: usize = 0b0010;
// /// If the reader count is zero: a writer is currently holding an exclusive lock.
// /// Otherwise: a writer is waiting for the remaining readers to exit the lock.
// const WRITER_BIT: usize = 0b0001;
// /// Mask of bits used to count readers.
// const READERS_MASK: usize = !0b1111;
// /// Base unit for counting readers.
// const ONE_READER: usize = 0b10000;
// 
// const TOKEN_HANDOFF: UnparkToken = UnparkToken(0b0001);
// const TOKEN_SHARED: ParkToken = ParkToken(ONE_READER);
// const TOKEN_EXCLUSIVE: ParkToken = ParkToken(WRITER_BIT);
// 
// #[inline(always)]
// fn with_thread_data<T>(f: impl FnOnce(&ThreadData) -> T) -> T {
//     let thread_data = ThreadData::new();
//     f(&thread_data)
// }
// 
// struct ThreadData {
//     parker: ThreadParker,
// 
//     // Linked list of parked threads in a bucket
//     next_in_queue: Cell<*const ThreadData>,
// 
//     // UnparkToken passed to this thread when it is unparked
//     unpark_token: Cell<UnparkToken>,
// 
//     // ParkToken value set by the thread when it was parked
//     park_token: Cell<ParkToken>,
// }
// 
// impl ThreadData {
//     fn new() -> ThreadData {
//         ThreadData {
//             parker: ThreadParker::new(),
//             next_in_queue: Cell::new(ptr::null()),
//             unpark_token: Cell::new(UnparkToken(0)),
//             park_token: Cell::new(ParkToken(0)),
//         }
//     }
// }
// 
// struct ParkingQueue {
//     // Lock protecting the queue
//     mutex: RawMutex,
// 
//     // Linked list of threads waiting on this bucket
//     queue_head: Cell<*const ParkingQueue>,
//     queue_tail: Cell<*const ParkingQueue>,
// 
//     // Next time at which point be_fair should be set
//     fair_timeout: FairTimeout,
// }
// 
// impl ParkingQueue {
//     fn new() -> Self {
//         let mutex = RawMutex::new();
//         let seed = &mutex as *const _ as u32;
//         Self {
//             mutex,
//             queue_head: Cell::new(ptr::null()),
//             queue_tail: Cell::new(ptr::null()),
//             fair_timeout: FairTimeout::new(Instant::now(), seed),
//         }
//     }
// }
// 
// #[inline]
// fn get_parking_queue(ptr: &AtomicPtr<ParkingQueue>) -> *const ParkingQueue {
//     let meta = ptr.load(Ordering::Acquire);
// 
//     // If there is no table, create one
//     if meta.is_null() {
//         create_parking_queue(ptr)
//     } else {
//         unsafe { &*meta }
//     }
// }
// 
// #[cold]
// fn create_parking_queue(ptr: &AtomicPtr<ParkingQueue>) -> *const ParkingQueue {
//     let new_parking_meta = Box::into_raw(Box::from(ParkingQueue::new()));
// 
//     // If this fails then it means some other thread created the hash table first.
//     let parking_meta = match ptr.compare_exchange(
//         ptr::null_mut(),
//         new_parking_meta,
//         Ordering::AcqRel,
//         Ordering::Acquire,
//     ) {
//         Ok(_) => new_parking_meta,
//         Err(old_table) => {
//             // Free the table we created
//             // SAFETY: `new_table` is created from `Box::into_raw` above and only freed here.
//             unsafe {
//                 let _ = Box::from_raw(new_parking_meta);
//             }
//             old_table
//         }
//     };
//     // SAFETY: The `HashTable` behind `table` is never freed. It is either the table pointer we
//     // created here, or it is one loaded from `HASHTABLE`.
//     unsafe { &*parking_meta }
// }
// 
// pub struct RawRwLock {
//     state: AtomicUsize,
//     parking_queue: AtomicPtr<ParkingQueue>,
// }
// 
// impl RawRwLock {
//     #[inline]
//     pub fn new() -> RawRwLock {
//         Self {
//             state: AtomicUsize::new(0),
//             parking_queue: AtomicPtr::new(ptr::null_mut()),
//         }
//     }
// 
//     #[inline]
//     pub fn lock_exclusive(&self) {
//         if self
//             .state
//             .compare_exchange_weak(0, WRITER_BIT, Ordering::Acquire, Ordering::Relaxed)
//             .is_err()
//         {
//             self.lock_exclusive_slow();
//         }
//     }
// 
//     #[inline]
//     pub fn try_lock_exclusive(&self) -> bool {
//         self.state
//             .compare_exchange(0, WRITER_BIT, Ordering::Acquire, Ordering::Relaxed)
//             .is_ok()
//     }
// 
//     #[inline]
//     pub fn lock_shared(&self) {
//         if !self.try_lock_shared_fast() {
//             self.lock_shared_slow();
//         }
//     }
// 
//     #[inline]
//     pub fn try_lock_shared(&self) -> bool {
//         if self.try_lock_shared_fast() {
//             true
//         } else {
//             self.try_lock_shared_slow()
//         }
//     }
// 
//     #[inline]
//     pub fn unlock_shared(&self) {
//         let state = self.state.fetch_sub(ONE_READER, Ordering::Release);
//         if state & (READERS_MASK | WRITER_PARKED_BIT) == (ONE_READER | WRITER_PARKED_BIT) {
//             self.unlock_shared_slow();
//         }
//     }
// 
//     #[cold]
//     fn lock_exclusive_slow(&self) {
//         let try_lock = |state: &mut usize| {
//             loop {
//                 if *state & WRITER_BIT != 0 {
//                     return false;
//                 }
// 
//                 // Grab WRITER_BIT if it isn't set, even if there are parked threads.
//                 match self.state.compare_exchange_weak(
//                     *state,
//                     *state | WRITER_BIT,
//                     Ordering::Acquire,
//                     Ordering::Relaxed,
//                 ) {
//                     Ok(_) => return true,
//                     Err(new_state) => *state = new_state,
//                 }
//             }
//         };
// 
//         // Step 1: grab exclusive ownership of WRITER_BIT
//         self.lock_common(TOKEN_EXCLUSIVE, try_lock, WRITER_BIT);
// 
//         // // Step 2: wait for all remaining readers to exit the lock.
//         // self.wait_for_readers();
//     }
// 
//     #[inline(always)]
//     fn try_lock_shared_fast(&self) -> bool {
//         let state = self.state.load(Ordering::Relaxed);
// 
//         // We can't allow grabbing a shared lock if there is a writer, even if
//         // the writer is still waiting for the remaining readers to exit.
//         if state & WRITER_BIT != 0 {
//             // To allow recursive locks, we make an exception and allow readers
//             // to skip ahead of a pending writer to avoid deadlocking, at the
//             // cost of breaking the fairness guarantees.
//             if state & READERS_MASK == 0 {
//                 return false;
//             }
//         }
// 
//         // Use hardware lock elision to avoid cache conflicts when multiple
//         // readers try to acquire the lock. We only do this if the lock is
//         // completely empty since elision handles conflicts poorly.
//         if let Some(new_state) = state.checked_add(ONE_READER) {
//             self.state
//                 .compare_exchange_weak(state, new_state, Ordering::Acquire, Ordering::Relaxed)
//                 .is_ok()
//         } else {
//             false
//         }
//     }
// 
//     #[cold]
//     fn try_lock_shared_slow(&self) -> bool {
//         let mut state = self.state.load(Ordering::Relaxed);
//         loop {
//             // This mirrors the condition in try_lock_shared_fast
//             #[allow(clippy::collapsible_if)]
//             if state & WRITER_BIT != 0 {
//                 if state & READERS_MASK == 0 {
//                     return false;
//                 }
//             }
//             match self.state.compare_exchange_weak(
//                 state,
//                 state
//                     .checked_add(ONE_READER)
//                     .expect("RwLock reader count overflow"),
//                 Ordering::Acquire,
//                 Ordering::Relaxed,
//             ) {
//                 Ok(_) => return true,
//                 Err(x) => state = x,
//             }
//         }
//     }
// 
//     #[cold]
//     fn lock_shared_slow(&self) {
//         let try_lock = |state: &mut usize| {
//             let mut spin_wait = SpinWait::new();
//             loop {
//                 // This is the same condition as try_lock_shared_fast
//                 #[allow(clippy::collapsible_if)]
//                 if *state & WRITER_BIT != 0 {
//                     if *state & READERS_MASK == 0 {
//                         return false;
//                     }
//                 }
// 
//                 if self
//                     .state
//                     .compare_exchange_weak(
//                         *state,
//                         state
//                             .checked_add(ONE_READER)
//                             .expect("RwLock reader count overflow"),
//                         Ordering::Acquire,
//                         Ordering::Relaxed,
//                     )
//                     .is_ok()
//                 {
//                     return true;
//                 }
// 
//                 // If there is high contention on the reader count then we want
//                 // to leave some time between attempts to acquire the lock to
//                 // let other threads make progress.
//                 spin_wait.spin_no_yield();
//                 *state = self.state.load(Ordering::Relaxed);
//             }
//         };
//         self.lock_common(TOKEN_SHARED, try_lock, WRITER_BIT);
//     }
// 
//     #[cold]
//     fn unlock_shared_slow(&self) {}
// 
//     #[inline]
//     fn lock_common(
//         &self,
//         token: ParkToken,
//         mut try_lock: impl FnMut(&mut usize) -> bool,
//         validate_flags: usize,
//     ) {
//         // let mut spin_wait = SpinWait::new();
//         // let mut state = self.state.load(Ordering::Relaxed);
//         // loop {
//         //     // Attempt to grab the lock
//         //     if try_lock(&mut state) {
//         //         return;
//         //     }
//         //
//         //     // If there are no parked threads, try spinning a few times.
//         //     if state & (PARKED_BIT | WRITER_PARKED_BIT) == 0 && spin_wait.spin() {
//         //         state = self.state.load(Ordering::Relaxed);
//         //         continue;
//         //     }
//         //
//         //     // Set the parked bit
//         //     if state & PARKED_BIT == 0
//         //         && let Err(new_state) = self.state.compare_exchange_weak(
//         //             state,
//         //             state | PARKED_BIT,
//         //             Ordering::Relaxed,
//         //             Ordering::Relaxed,
//         //         )
//         //     {
//         //         state = new_state;
//         //         continue;
//         //     }
//         //
//         //     let parking_meta = unsafe { &(*get_parking_queue(&self.parking_queue)) };
//         //     parking_meta.mutex.lock();
//         //     state = self.state.load(Ordering::Relaxed);
//         //     if state & PARKED_BIT != 0 && (state & validate_flags != 0) {
//         //         parking_meta.mutex.unlock();
//         //         return;
//         //     }
//         //     parking_meta.mutex.unlock();
//         //
//         //     match parking_meta.queue.internal_enqueue(
//         //         parking_meta.queue.state.load(Ordering::Relaxed),
//         //         |park_token| {
//         //             park_token.set(token);
//         //         },
//         //     ) {
//         //         // The thread that unparked us passed the lock on to us
//         //         // directly without unlocking it.
//         //         ParkResult::Unparked(TOKEN_HANDOFF) => return,
//         //
//         //         // We were unparked normally, try acquiring the lock again
//         //         ParkResult::Unparked(_) => (),
//         //
//         //         // The validation function failed, try locking again
//         //         ParkResult::Invalid => (),
//         //     }
//         //     // Loop back and try locking again
//         //     spin_wait.reset();
//         //     state = self.state.load(Ordering::Relaxed);
//         // }
//     }
// }
