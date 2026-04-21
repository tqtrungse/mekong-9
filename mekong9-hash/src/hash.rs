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

use core::hash::Hasher;

#[cfg(target_pointer_width = "32")]
use crate::internal::low_level::{combine_contiguous_on32bit, combine_raw};
#[cfg(target_pointer_width = "64")]
use crate::internal::low_level::{combine_contiguous_on64bit, combine_raw};
use crate::internal::unaligned_access::{unaligned_load16, unaligned_load32, unaligned_load64};

static K_SEED: u8 = 0;

#[derive(Clone)]
pub struct MixingHasher {
    state: u64,
}

impl MixingHasher {
    #[inline]
    pub fn with_state(state: u64) -> Self {
        Self { state }
    }
}

impl Default for MixingHasher {
    #[inline]
    fn default() -> Self {
        Self::with_state(*(&K_SEED) as u64)
    }
}

impl Hasher for MixingHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.state
    }

    #[cfg(target_pointer_width = "64")]
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state = unsafe { combine_contiguous_on64bit(self.state, bytes) };
    }

    #[cfg(target_pointer_width = "32")]
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state = combine_contiguous_on32bit(self.state, bytes);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        if i == 1 {
            // for bool = true
            self.state = combine_raw(self.state, usize::MAX as u64);
        } else {
            self.state = combine_raw(self.state, i as u64);
        }
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        let value = unaligned_load16(&i as *const u16 as *const u8);
        self.state = combine_raw(self.state, value as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        let value = unaligned_load32(&i as *const u32 as *const u8);
        self.state = combine_raw(self.state, value as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        let value = unaligned_load64(&i as *const u64 as *const u8);
        self.state = combine_raw(self.state, value);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.write_u64(i as u64);
        self.write_u64((i >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(self.state.wrapping_add(0x57).wrapping_add(i as u64));
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.state = combine_raw(self.state, i as u64);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        let value = unaligned_load16(&i as *const i16 as *const u8);
        self.state = combine_raw(self.state, value as u64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        let value = unaligned_load32(&i as *const i32 as *const u8);
        self.state = combine_raw(self.state, value as u64);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        let value = unaligned_load64(&i as *const i64 as *const u8);
        self.state = combine_raw(self.state, value);
    }

    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.write_u64(i as u64);
        self.write_u64((i >> 64) as u64);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        if size_of::<isize>() == 4 {
            self.write_i32(i as i32);
        } else {
            self.write_i64(i as i64);
        }
    }
}

#[cfg(test)]
mod test {
    use core::hash::{Hash, Hasher};
    use std::ffi::CString;
    use std::rc::Rc;
    use std::sync::Arc;

    use crate::MixingHasher;

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = MixingHasher::default();
        v.hash(&mut h);
        h.finish()
    }

    fn verify_type_implements_hash_correctly<T>(values: &[T]) -> Result<(), String>
    where
        T: Hash + Eq,
    {
        // 1. Equal ⇒ same hash
        for i in 0..values.len() {
            for j in 0..values.len() {
                if values[i] == values[j] {
                    let hi = hash_of(&values[i]);
                    let hj = hash_of(&values[j]);
                    if hi != hj {
                        return Err(format!(
                            "Equal values have different hashes: {:?} vs {:?}",
                            hi, hj
                        ));
                    }
                }
            }
        }

        // 2. Hash stability
        for v in values {
            let h1 = hash_of(v);
            let h2 = hash_of(v);
            if h1 != h2 {
                return Err("Hash is not stable".into());
            }
        }

        Ok(())
    }

    #[test]
    fn test_bool() {
        match verify_type_implements_hash_correctly(&[true, false]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }
    }

    #[test]
    fn test_hash_consistent_across_int_types() {
        let expected = hash_of(&1i8);
        assert_eq!(expected, hash_of(&1i8));
        assert_eq!(expected, hash_of(&1u16));
        assert_eq!(expected, hash_of(&1u32));
        assert_eq!(expected, hash_of(&1u64));
    }

    #[test]
    fn test_pointer() {
        let i: u32 = 1;
        let ptr: *const u32 = &i;
        let null_ptr: *const u32 = std::ptr::null();
        let mut_null_ptr: *mut u32 = std::ptr::null_mut();

        match verify_type_implements_hash_correctly(&[
            &i as *const u32,
            ptr,
            null_ptr,
            mut_null_ptr,
        ]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }
    }

    #[test]
    fn test_string() {
        match verify_type_implements_hash_correctly(&[
            String::default(),
            String::from(""),
            String::from("foo"),
            String::from("foofoo"),
            "x".repeat(2048),
            "a".repeat(5000),
        ]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }

        match verify_type_implements_hash_correctly(&[
            CString::default(),
            CString::new("").unwrap(),
            CString::new("foo").unwrap(),
            CString::new("foofoo").unwrap(),
            CString::new("x".repeat(2048)).unwrap(),
            CString::new("a".repeat(5000)).unwrap(),
        ]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }

        match verify_type_implements_hash_correctly(&[
            "",
            "foo",
            "foofoo",
            "x".repeat(2048).as_str(),
            "a".repeat(5000).as_str(),
        ]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }
    }

    #[test]
    fn test_array_slice() {
        match verify_type_implements_hash_correctly(&[[1, 2, 3], [0, 23, 42]]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }

        match verify_type_implements_hash_correctly(&[
            Vec::<i32>::default().as_slice(),
            Vec::from([0, 23, 42]).as_slice(),
        ]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }

        match verify_type_implements_hash_correctly(&[
            Vec::<i32>::default(),
            Vec::from([0, 23, 42]),
        ]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }
    }

    #[test]
    fn test_wrapped_type() {
        match verify_type_implements_hash_correctly(&[Some(1), None]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }

        match verify_type_implements_hash_correctly(&[Box::new(1), Box::new(1), Box::new(2)]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }

        let rc = Rc::new(1);
        match verify_type_implements_hash_correctly(&[rc.clone(), Rc::new(1), Rc::new(2)]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }

        let arc = Arc::new(1);
        match verify_type_implements_hash_correctly(&[
            arc.clone(),
            arc.clone(),
            Arc::new(1),
            Arc::new(2),
        ]) {
            Ok(_) => {}
            Err(e) => panic!("{}", e),
        }
    }
}
