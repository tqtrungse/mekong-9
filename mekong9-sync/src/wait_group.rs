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

use core::sync::atomic::{AtomicIsize, Ordering};

use crate::internal::ParkingQueue;

/// A [`WaitGroup`] waits for a collection of threads to finish.
/// The main thread calls [`add`] to set the number of threads to wait for.
/// Then each of the threads runs and calls [`done`] when finished.
/// At the same time, [`wait`] can be used to block until all threads
/// have finished.
///
/// A [`WaitGroup`] must not be copied after first use.
///
/// A call to [`done`] “synchronizes before” the return of any
/// Wait call that it unblocks.
///
/// [`add`]: WaitGroup::add
/// [`done`]: WaitGroup::done
/// [`wait`]: WaitGroup::wait
pub struct WaitGroup {
    workers: AtomicIsize,
    parking_queue: ParkingQueue,
}

impl WaitGroup {
    /// Creates a new [`WaitGroup`] with number member of a group.
    ///
    /// [`WaitGroup`]: WaitGroup
    #[inline]
    pub fn with_size(n: usize) -> Self {
        Self {
            workers: AtomicIsize::new(n as isize),
            parking_queue: ParkingQueue::new(),
        }
    }

    /// Adds delta, which may be negative, to the [`WaitGroup`] counter.
    /// If the counter becomes zero, all threads blocked on [`wait`] are released.
    /// If the counter goes negative, [`add`] panics.
    ///
    /// Note that calls with a positive delta that occur when the counter is zero
    /// must happen before a [`wait`]. Calls with a negative delta, or calls with a
    /// positive delta that start when the counter is greater than zero, may happen
    /// at any time.
    /// Typically, this means the calls to [`add`] should execute before the statement
    /// creating the thread or other event to be waited for.
    ///
    /// If a [`WaitGroup`] is reused to wait for several independent sets of events,
    /// new [`add`] calls must happen after all previous [`wait`] calls have returned.
    ///
    /// Example:
    ///
    /// ```
    /// use mekong9_sync::WaitGroup;
    ///
    /// let wg = std::sync::Arc::new(WaitGroup::with_size(1));
    /// let wg_clone = wg.clone();
    ///
    /// let count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    /// let count_clone = count.clone();
    ///
    /// let thread = std::thread::spawn(move || {
    ///     wg_clone.add(1);
    ///     count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ///     wg_clone.done();
    ///     wg_clone.wait();
    ///
    ///     assert_eq!(count_clone.load(std::sync::atomic::Ordering::Relaxed), 2);
    /// });
    ///
    /// count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    /// wg.done();
    /// wg.wait();
    ///
    /// thread.join().unwrap();
    /// assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 2);
    /// ```
    ///
    /// [`WaitGroup`]: WaitGroup
    /// [`wait`]: WaitGroup::wait
    #[inline]
    pub fn add(&self, n: usize) {
        self.workers.fetch_add(n as isize, Ordering::Relaxed);
    }

    /// Decrements the [`WaitGroup`] counter by one.
    ///
    /// Example see [`add`]
    ///
    /// [`WaitGroup`]: WaitGroup
    /// [`add`]: WaitGroup::add
    #[inline]
    pub fn done(&self) {
        let workers = self.workers.fetch_sub(1, Ordering::AcqRel);
        assert!(workers >= 1);
        if workers == 1 {
            self.parking_queue.unpark_all();
        }
    }

    /// Blocks until the [`WaitGroup`] counter is zero.
    ///
    /// Example see [`add`]
    ///
    /// [`WaitGroup`]: WaitGroup
    /// [`add`]: WaitGroup::add
    pub fn wait(&self) {
        self.parking_queue
            .park(|| self.workers.load(Ordering::Acquire) == 0);
    }
}

unsafe impl Send for WaitGroup {}
unsafe impl Sync for WaitGroup {}

#[cfg(test)]
mod tests {
    use std::panic;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::mpsc::sync_channel;
    use std::sync::Arc;

    use crate::WaitGroup;

    #[test]
    fn test_wait_group() {
        let wg1 = Arc::new(WaitGroup::with_size(0));
        let wg2 = Arc::new(WaitGroup::with_size(0));

        for _ in 0..8 {
            test_impl(&wg1, &wg2);
        }
    }

    #[test]
    fn test_done_without_size() {
        let result = std::panic::catch_unwind(|| {
            let wg = crate::WaitGroup::with_size(0);
            wg.done();
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_counter_panic() {
        let result = panic::catch_unwind(|| {
            let wg = WaitGroup::with_size(0);
            wg.add(1);
            wg.done();
            wg.done();
        });

        match result {
            Ok(_) => panic!("Should panic"),
            Err(err) => {
                let msg = if let Some(s) = err.downcast_ref::<&'static str>() {
                    *s
                } else if let Some(s) = err.downcast_ref::<String>() {
                    s.as_str()
                } else {
                    "Unknown panic payload"
                };
                if msg != "assertion failed: workers >= 1" {
                    panic!("Unexpected panic: {:?}", msg);
                }
            }
        }
    }

    #[test]
    fn test_race() {
        for _ in 0..3000 {
            let wg = Arc::new(WaitGroup::with_size(0));
            let n = Arc::new(AtomicU32::new(0));

            let wg_clone_1 = wg.clone();
            let n_clone_1 = n.clone();
            wg.add(1);
            let thread_1 = std::thread::spawn(move || {
                n_clone_1.fetch_add(1, Ordering::Relaxed);
                wg_clone_1.done();
            });

            let wg_clone_2 = wg.clone();
            let n_clone_2 = n.clone();
            wg.add(1);
            let thread_2 = std::thread::spawn(move || {
                n_clone_2.fetch_add(1, Ordering::Relaxed);
                wg_clone_2.done();
            });

            let wg_clone_3 = wg.clone();
            let n_clone_3 = n.clone();
            wg.add(1);
            let thread_3 = std::thread::spawn(move || {
                n_clone_3.fetch_add(1, Ordering::Relaxed);
                wg_clone_3.done();
            });

            let wg_clone_4 = wg.clone();
            let n_clone_4 = n.clone();
            wg.add(1);
            let thread_4 = std::thread::spawn(move || {
                n_clone_4.fetch_add(1, Ordering::Relaxed);
                wg_clone_4.done();
            });

            wg.wait();
            thread_1.join().unwrap();
            thread_2.join().unwrap();
            thread_3.join().unwrap();
            thread_4.join().unwrap();

            assert_eq!(n.load(Ordering::Relaxed), 4);
        }
    }

    #[test]
    fn test_wait_group_stress() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::thread;

        let iterations = 3000;
        let num_workers = 12;
        let num_waiters = 4;

        for _ in 0..iterations {
            let wg = Arc::new(WaitGroup::with_size(0));
            let counter = Arc::new(AtomicU32::new(0));

            wg.add(num_workers as usize);

            // 1. Spawning Waiters
            let mut waiter_handles = vec![];
            for _ in 0..num_waiters {
                let wg_c = wg.clone();
                let counter_c = counter.clone();
                waiter_handles.push(thread::spawn(move || {
                    wg_c.wait();
                    assert_eq!(counter_c.load(Ordering::Acquire), num_workers);
                }));
            }

            // 2. Spawning Workers
            let mut worker_handles = vec![];
            for _ in 0..num_workers {
                let wg_c = wg.clone();
                let counter_c = counter.clone();
                worker_handles.push(thread::spawn(move || {
                    // Add a little random delay so that threads calling done() are close together.
                    thread::yield_now();
                    counter_c.fetch_add(1, Ordering::Release);
                    wg_c.done();
                }));
            }

            for h in worker_handles {
                h.join().unwrap();
            }
            for h in waiter_handles {
                h.join().unwrap();
            }

            assert_eq!(counter.load(Ordering::Relaxed), num_workers);
        }
    }

    fn test_impl(wg1: &Arc<WaitGroup>, wg2: &Arc<WaitGroup>) {
        let n = 16;
        wg1.add(n);
        wg2.add(n);

        let (tx, rx) = sync_channel(n);
        for _ in 0..n {
            let wg1_clone = wg1.clone();
            let wg2_clone = wg2.clone();
            let tx_clone = tx.clone();
            std::thread::spawn(move || {
                wg1_clone.done();
                wg2_clone.wait();
                tx_clone.send(true).unwrap();
            });
        }
        wg1.wait();
        for _ in 0..n {
            if rx.try_recv().is_ok() {
                panic!("WaitGroup released group too soon");
            }
            wg2.done();
        }
        for _ in 0..n {
            rx.recv().unwrap();
        }
    }
}
