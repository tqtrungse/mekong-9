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

//! Edit from parking_lot::word_lock.

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};

use crate::internal::RawMutex;

pub struct Mutex<T: ?Sized> {
    raw: RawMutex,
    data: UnsafeCell<T>,
}

unsafe impl<T: ?Sized + Send> Send for Mutex<T> {}

unsafe impl<T: ?Sized + Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    #[inline]
    pub fn new(data: T) -> Self {
        Self {
            raw: RawMutex::new(),
            data: UnsafeCell::new(data),
        }
    }

    /// Consumes this mutex, returning the underlying data.
    #[inline]
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }
}

impl<T: ?Sized> Mutex<T> {
    /// Acquires a mutex, blocking the current thread until it is able to do so.
    ///
    /// This function will block the local thread until it is available to acquire
    /// the mutex. Upon returning, the thread is the only thread with the mutex
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the mutex will be unlocked.
    ///
    /// Attempts to lock a mutex in the thread which already holds the lock will
    /// result in a deadlock.
    #[inline]
    #[track_caller]
    pub fn lock(&self) -> MutexGuard<'_, T> {
        self.raw.lock();
        MutexGuard { parent: self }
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    #[inline]
    #[track_caller]
    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        if self.raw.try_lock() {
            return Some(MutexGuard { parent: self });
        }
        None
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this call borrows the `Mutex` mutably, no actual locking needs to
    /// take place---the mutable borrow statically guarantees no locks exist.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.data.get() }
    }

    /// Checks whether the mutex is currently locked.
    #[inline]
    #[track_caller]
    pub fn is_locked(&self) -> bool {
        let acquired_lock = self.raw.try_lock();
        if acquired_lock {
            self.raw.unlock();
        }
        !acquired_lock
    }

    /// Returns a raw pointer to the underlying data.
    ///
    /// This is useful when combined with `mem::forget` to hold a lock without
    /// the need to maintain a `MutexGuard` object alive, for example when
    /// dealing with FFI.
    ///
    /// # Safety
    ///
    /// You must ensure that there are no data races when dereferencing the
    /// returned pointer, for example if the current thread logically owns
    /// a `MutexGuard` but that guard has been discarded using `mem::forget`.
    #[inline]
    pub fn data_ptr(&self) -> *mut T {
        self.data.get()
    }
}

/// Not fair unlock.
///
/// In other words, it means that anyone can hold the lock.
/// Those who have been waiting a long time may have to wait even longer
/// because of the arrivals of newcomers.
pub struct MutexGuard<'a, T: ?Sized> {
    parent: &'a Mutex<T>,
}

impl<T: ?Sized> Drop for MutexGuard<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {
        self.parent.raw.unlock();
    }
}

impl<T: ?Sized> Deref for MutexGuard<'_, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        unsafe { &*self.parent.data.get() }
    }
}

impl<T: ?Sized> DerefMut for MutexGuard<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.parent.data.get() }
    }
}

/// Fair unlock.
///
/// Those who have been waiting a long time will be given priority to lock.
pub struct MutexGuardFair<'a, T: ?Sized> {
    parent: &'a Mutex<T>,
}

impl<T: ?Sized> Drop for MutexGuardFair<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {
        self.parent.raw.unlock_fair();
    }
}

impl<T: ?Sized> Deref for MutexGuardFair<'_, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        unsafe { &*self.parent.data.get() }
    }
}

impl<T: ?Sized> DerefMut for MutexGuardFair<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.parent.data.get() }
    }
}

#[cfg(test)]
mod tests {
    use crate::Mutex;
    // use std::collections::HashMap;
    // use std::ops::Deref;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::ops::AddAssign;
    use std::sync::Arc;
    use std::sync::mpsc::channel;
    use std::thread;

    // struct Packet<T>(Arc<(Mutex<T>, Condvar)>);

    #[derive(Eq, PartialEq, Debug)]
    struct NonCopy(i32);

    // unsafe impl<T: Send> Send for Packet<T> {}
    // unsafe impl<T> Sync for Packet<T> {}

    #[test]
    fn smoke() {
        let m = Mutex::new(());
        drop(m.lock());
        drop(m.lock());
    }

    #[test]
    fn lots_and_lots() {
        const J: u32 = 5000;
        const K: u32 = 300;

        let m = Arc::new(Mutex::new(0));

        fn inc(m: &Mutex<u32>) {
            for _ in 0..J {
                *m.lock() += 1;
            }
        }

        let (tx, rx) = channel();
        for _ in 0..K {
            let tx2 = tx.clone();
            let m2 = m.clone();
            thread::spawn(move || {
                inc(&m2);
                tx2.send(()).unwrap();
            });
            let tx2 = tx.clone();
            let m2 = m.clone();
            thread::spawn(move || {
                inc(&m2);
                tx2.send(()).unwrap();
            });
        }

        drop(tx);
        for _ in 0..2 * K {
            rx.recv().unwrap();
        }
        assert_eq!(*m.lock(), J * K * 2);
    }

    #[test]
    fn try_lock() {
        let m = Mutex::new(());
        *m.try_lock().unwrap() = ();
    }

    #[test]
    fn test_into_inner() {
        let m = Mutex::new(NonCopy(10));
        assert_eq!(m.into_inner(), NonCopy(10));
    }

    #[test]
    fn test_into_inner_drop() {
        struct Foo(Arc<AtomicUsize>);
        impl Drop for Foo {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let num_drops = Arc::new(AtomicUsize::new(0));
        let m = Mutex::new(Foo(num_drops.clone()));
        assert_eq!(num_drops.load(Ordering::SeqCst), 0);
        {
            let _inner = m.into_inner();
            assert_eq!(num_drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(num_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_get_mut() {
        let mut m = Mutex::new(NonCopy(10));
        *m.get_mut() = NonCopy(20);
        assert_eq!(m.into_inner(), NonCopy(20));
    }

    // #[test]
    // fn test_mutex_arc_condvar() {
    //     let packet = Packet(Arc::new((Mutex::new(false), Condvar::new())));
    //     let packet2 = Packet(packet.0.clone());
    //     let (tx, rx) = channel();
    //     let _t = thread::spawn(move || {
    //         // wait until parent gets in
    //         rx.recv().unwrap();
    //         let (lock, cvar) = &*packet2.0;
    //         let mut lock = lock.lock();
    //         *lock = true;
    //         cvar.notify_one();
    //     });
    //
    //     let (lock, cvar) = &*packet.0;
    //     let mut lock = lock.lock();
    //     tx.send(()).unwrap();
    //     assert!(!*lock);
    //     while !*lock {
    //         cvar.wait(&mut lock);
    //     }
    // }

    #[test]
    fn test_mutex_arc_nested() {
        // Tests nested mutexes and access
        // to underlying data.
        let arc = Arc::new(Mutex::new(1));
        let arc2 = Arc::new(Mutex::new(arc));
        let (tx, rx) = channel();
        let _t = thread::spawn(move || {
            let lock = arc2.lock();
            let lock2 = lock.lock();
            assert_eq!(*lock2, 1);
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();
    }

    #[test]
    fn test_mutex_arc_access_in_unwind() {
        let arc = Arc::new(Mutex::new(1));
        let arc2 = arc.clone();
        let _ = thread::spawn(move || {
            struct Unwinder {
                i: Arc<Mutex<i32>>,
            }
            impl Drop for Unwinder {
                fn drop(&mut self) {
                    *self.i.lock() += 1;
                }
            }
            let _u = Unwinder { i: arc2 };
            panic!();
        })
        .join();
        let lock = arc.lock();
        assert_eq!(*lock, 2);
    }

    #[test]
    fn test_mutex_unsized() {
        let mutex: &Mutex<[i32]> = &Mutex::new([1, 2, 3]);
        {
            let b = &mut *mutex.lock();
            b[0] = 4;
            b[2] = 5;
        }
        let comp: &[i32] = &[4, 2, 5];
        assert_eq!(&*mutex.lock(), comp);
    }

    #[test]
    fn test_mutexguard_sync() {
        fn sync<T: Sync>(_: T) {}

        let mutex = Mutex::new(());
        sync(mutex.lock());
    }

    // #[test]
    // fn test_mutex_debug() {
    //     let mutex = Mutex::new(vec![0u8, 10]);
    //
    //     assert_eq!(format!("{:?}", mutex), "Mutex { data: [0, 10] }");
    //     let _lock = mutex.lock();
    //     assert_eq!(format!("{:?}", mutex), "Mutex { data: <locked> }");
    // }

    // #[test]
    // fn test_map_or_err_not_mapped() {
    //     let mut map = HashMap::new();
    //     map.insert("hello".to_string(), "world".to_string());
    //
    //     let mutex = Mutex::new(map);
    //     let guard = mutex.lock();
    //     let guard = match MutexGuard::try_map_or_err(guard, |the_map| {
    //         the_map.get_mut("hello2").ok_or(12345i32)
    //     }) {
    //         Ok(_) => unreachable!(),
    //         Err((guard, data)) => {
    //             assert_eq!(data, 12345i32);
    //             assert_eq!(guard.get("hello"), Some(&"world".to_string()));
    //             guard
    //         }
    //     };
    //
    //     // Lets try again
    //     let mapped_guard = match MutexGuard::try_map_or_err(guard, |the_map| {
    //         the_map.get_mut("hello").ok_or("unreachable")
    //     }) {
    //         Ok(mapped_guard) => mapped_guard,
    //         Err((_, _)) => unreachable!(),
    //     };
    //
    //     assert_eq!(mapped_guard.as_str(), "world");
    //
    //     match MappedMutexGuard::try_map_or_err(mapped_guard, |the_string| {
    //         if the_string != "world" {
    //             //unreachable
    //             Ok(the_string.as_mut_str())
    //         } else {
    //             Err(45678i32)
    //         }
    //     }) {
    //         Ok(_) => unreachable!(),
    //         Err((guard, err)) => {
    //             assert_eq!(guard.as_str(), "world");
    //             assert_eq!(err, 45678i32);
    //         }
    //     };
    // }
    //
    // #[test]
    // fn test_map_or_err_mapped() {
    //     let mut map = HashMap::new();
    //     map.insert("hello".to_string(), "world".to_string());
    //
    //     let mutex = Mutex::new(map);
    //     let guard = mutex.lock();
    //     let mapped_guard = match MutexGuard::try_map_or_err(guard, |the_map| {
    //         the_map.get_mut("hello").ok_or("unreachable")
    //     }) {
    //         Ok(mapped_guard) => mapped_guard,
    //         Err((_, _)) => unreachable!(),
    //     };
    //
    //     assert_eq!(mapped_guard.as_str(), "world");
    //
    //     match MappedMutexGuard::try_map_or_err(mapped_guard, |the_string| {
    //         if the_string == "world" {
    //             Ok(the_string.as_mut_str())
    //         } else {
    //             Err("unreachable")
    //         }
    //     }) {
    //         Ok(mapped_guard) => assert_eq!(mapped_guard.deref(), "world"),
    //         Err((_, _)) => unreachable!(),
    //     };
    // }
}
