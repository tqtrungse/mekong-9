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

//! Edit from parking_lot.

use core::cell::Cell;
use std::time::{Duration, Instant};

pub struct FairTimeout {
    // Next time at which point be_fair should be set
    timeout: Cell<Instant>,

    // the PRNG state for calculating the next timeout
    seed: Cell<u32>,
}

impl FairTimeout {
    #[inline]
    pub fn new(timeout: Instant, seed: u32) -> FairTimeout {
        FairTimeout { 
            timeout: Cell::new(timeout), 
            seed: Cell::new(seed), 
        }
    }

    // Determine whether we should force a fair unlock, and update the timeout
    #[inline]
    pub fn is_timeout(&self) -> bool {
        let now = Instant::now();
        if now > self.timeout.get() {
            // Time between 0 and 1ms.
            let nanos = self.gen_u32() % 1_000_000;
            self.timeout.set(now + Duration::new(0, nanos));
            true
        } else {
            false
        }
    }

    // Pseudorandom number generator from the "Xorshift RNGs" paper by George Marsaglia.
    fn gen_u32(&self) -> u32 {
        let mut seed = self.seed.get();
        
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        
        self.seed.set(seed);
        seed
    }
}