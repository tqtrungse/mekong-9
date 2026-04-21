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

#[cfg(all(target_feature = "sse2", target_arch = "x86_64", not(miri)))]
use core::arch::x86_64::{
    __m128i, _mm_add_epi64, _mm_aesdec_si128, _mm_aesenc_si128, _mm_crc32_u32, _mm_crc32_u64,
    _mm_crc32_u8, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_loadu_si128, _mm_set_epi64x,
    _mm_sub_epi64,
};

use mekong9_utils::{likely, unlikely};

use crate::internal::city;
use crate::internal::prefetch::prefetch_to_local_cache;
use crate::internal::unaligned_access::{unaligned_load32, unaligned_load64};

// ARM 32-bit - default to 64, could be refined based on specific ARM version
#[cfg_attr(
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm",
    ),
    repr(align(64))
)]
// Default fallback
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "powerpc64",
        target_arch = "aarch64",
        target_arch = "arm"
    )),
    repr(align(64))
)]
#[cfg_attr(target_arch = "powerpc64", repr(align(128)))]
struct CachePadded<T>(pub T);

const K_MUL: u64 = 0x79d5f9e0de1e8cf5;
const CACHELINE_SIZE: usize = size_of::<CachePadded<u8>>();
const PIECEWISE_CHUNK_SIZE: usize = 1024;

static K_STATIC_RANDOM_DATA: CachePadded<[u64; 5]> = CachePadded([
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
    0x4528_21e6_38d0_1377,
]);

// Implementation of the base case for combine_contiguous where we actually
// mix the bytes into the state.
// Dispatch to different implementations of combine_contiguous depending
// on the value of `sizeof(size_t)`.
#[inline]
pub(crate) fn combine_contiguous_on32bit(state: u64, first: &[u8]) -> u64 {
    let len = first.len();
    // For large values we use CityHash, for small ones we use custom low latency
    // hash.
    if len <= 8 {
        return combine_small_contiguous(precombine_length_mix(state, len), first);
    }
    combine_large_contiguous_on32bit_length_gt8(state, first)
}

#[cfg(all(target_feature = "sse2", target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "aes")]
pub(crate) fn combine_contiguous_on64bit(state: u64, first: &[u8]) -> u64 {
    let len = first.len();

    if likely(is_x86_feature_detected!("sse4.2")) {
        unsafe {
            if unlikely(len > 32) {
                return combine_large_contiguous_on64bit_length_gt32(state, first);
            }
            // `mul` is the salt that is used for final mixing. It is important to fill
            // high 32 bits because CRC wipes out high 32 bits.
            // `rotate_right` is important to mix `len` into high 32 bits.
            let mut mul = K_MUL.rotate_right(len as u32);
            // Only low 32 bits of each uint64_t are used in CRC32 so we use swap_bytes to
            // move high 32 bits to low 32 bits. It has slightly smaller binary size than
            // `>> 32`. `state + 8 * len` is a single instruction on both x86 and ARM, so
            // we use it to better mix length. Although only the low 32 bits of the pair
            // elements are used, we use pair<uint64_t, uint64_t> for better generated
            // code.
            let mut crcs = (
                state.wrapping_add(8).wrapping_mul(len as u64),
                state.swap_bytes(),
            );

            // All CRC operations here directly read bytes from the memory.
            // Single fused instructions are used, like `crc32 rcx, qword ptr [rsi]`.
            // On x86, llvm-mca reports latency `R + 2` for such fused instructions, while
            // `R + 3` for two separate `mov` + `crc` instructions. `R` is the latency of
            // reading the memory. Fused instructions also reduce register pressure
            // allowing surrounding code to be more efficient when this code is inlined.
            if len > 8 {
                crcs = (
                    _mm_crc32_u64(crcs.0, read8(first)),
                    _mm_crc32_u64(crcs.1, read8(&first[len - 8..])),
                );
                if len > 16 {
                    // We compute the second round of dependent CRC32 operations.
                    crcs = (
                        _mm_crc32_u64(crcs.0, read8(&first[len - 16..])),
                        _mm_crc32_u64(crcs.1, read8(&first[8..])),
                    );
                }
            } else {
                if len >= 4 {
                    // We use CRC for 4 bytes to benefit from the fused instruction and better
                    // hash quality.
                    // Using `xor` or `add` may reduce latency for this case, but would
                    // require more registers, more instructions and will have worse hash
                    // quality.
                    crcs = (
                        _mm_crc32_u32(crcs.0 as u32, read4(first)) as u64,
                        _mm_crc32_u32(crcs.1 as u32, read4(&first[len - 4..])) as u64,
                    );
                } else if len >= 1 {
                    // We mix three bytes all into different output registers.
                    // This way, we do not need shifting of these bytes (so they don't overlap
                    // with each other).
                    crcs = (
                        _mm_crc32_u8(crcs.0 as u32, first[0]) as u64,
                        _mm_crc32_u8(crcs.1 as u32, first[len - 1]) as u64,
                    );
                    // Middle byte is mixed weaker. It is a new byte only for len == 3.
                    // Mixing is independent of CRC operations so it is scheduled ASAP.
                    mul = mul.wrapping_add(first[len / 2] as u64);
                }
            }
            // `mul` is mixed into both sides of `Mix` to guarantee non-zero values for
            // both multiplicands. Using Mix instead of just multiplication here improves
            // hash quality, especially for short strings.
            return mix(mul.wrapping_sub(crcs.0), crcs.1.wrapping_sub(mul));
        }
    }

    if len <= 8 {
        return combine_small_contiguous(precombine_length_mix(state, len), first);
    }
    if len <= 16 {
        return combine_contiguous_9to16(precombine_length_mix(state, len), first);
    }
    if len <= 32 {
        return combine_contiguous_17to32(precombine_length_mix(state, len), first);
    }
    // We must not mix length into the state here because calling
    // CombineContiguousImpl twice with PiecewiseChunkSize() must be equivalent
    // to calling CombineLargeContiguousImpl once with 2 * PiecewiseChunkSize().
    combine_large_contiguous_on64bit_length_gt32(state, first)
}

#[cfg(not(all(target_feature = "sse2", target_arch = "x86_64", not(miri))))]
pub(crate) fn combine_contiguous_on64bit(state: u64, first: &[u8]) -> u64 {
    let len = first.len();
    // For large values we use LowLevelHash or CityHash depending on the platform,
    // for small ones we use custom low latency hash.
    if len <= 8 {
        return combine_small_contiguous(precombine_length_mix(state, len), first);
    }
    if len <= 16 {
        return combine_contiguous_9to16(precombine_length_mix(state, len), first);
    }
    if len <= 32 {
        return combine_contiguous_17to32(precombine_length_mix(state, len), first);
    }
    // We must not mix length into the state here because calling
    // CombineContiguousImpl twice with PiecewiseChunkSize() must be equivalent
    // to calling CombineLargeContiguousImpl once with 2 * PiecewiseChunkSize().
    combine_large_contiguous_on64bit_length_gt32(state, first)
}

#[inline]
fn prefetch_future_data_to_local_cache(ptr: &[u8]) {
    let dist = 5 * CACHELINE_SIZE;
    if ptr.len() < dist {
        return;
    }
    prefetch_to_local_cache(ptr[5 * CACHELINE_SIZE..].as_ptr());
}

#[inline]
fn mix(lhs: u64, rhs: u64) -> u64 {
    let mut m = lhs as u128;
    m = m.wrapping_mul(rhs as u128);
    ((m & 0xFFFF_FFFF_FFFF_FFFF) as u64) ^ ((m >> 64) as u64)
}

#[cfg(not(all(target_feature = "sse2", target_arch = "x86_64", not(miri))))]
fn mix_32_bytes(ptr: &[u8], current_state: u64) -> u64 {
    let a = unaligned_load64(ptr.as_ptr());
    let b = unaligned_load64(ptr[8..].as_ptr());
    let c = unaligned_load64(ptr[16..].as_ptr());
    let d = unaligned_load64(ptr[24..].as_ptr());

    let cs_0 = mix(a ^ K_STATIC_RANDOM_DATA.0[1], b ^ current_state);
    let cs_1 = mix(c ^ K_STATIC_RANDOM_DATA.0[2], d ^ current_state);
    cs_0 ^ cs_1
}

#[cfg(all(target_feature = "sse2", target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "aes")]
#[inline]
unsafe fn mix4x16vectors(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> u64 {
    // res128 = encrypt(a + c, d) + decrypt(b - d, a)
    let res128 = _mm_add_epi64(
        _mm_aesenc_si128(_mm_add_epi64(a, c), d),
        _mm_aesdec_si128(_mm_sub_epi64(b, d), a),
    );

    let x64 = _mm_cvtsi128_si64(res128) as u64;
    let y64 = _mm_extract_epi64(res128, 1) as u64;
    x64 ^ y64
}

#[cfg(all(target_feature = "sse2", target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "aes")]
#[inline]
fn read4(p: &[u8]) -> u32 {
    unaligned_load32(p.as_ptr())
}

#[inline]
fn read8(p: &[u8]) -> u64 {
    unaligned_load64(p.as_ptr())
}

// Reads 9 to 16 bytes from p.
// The first 8 bytes are in .first, and the rest of the bytes are in .second
// along with duplicated bytes from .first if len<16.
#[inline]
fn read9to16(p: &[u8]) -> (u64, u64) {
    (read8(p), read8(&p[p.len() - 8..]))
}

// Reads 4 to 8 bytes from p.
// Bytes are permuted and some input bytes may be duplicated in output.
#[inline]
fn read4to8(p: &[u8]) -> u64 {
    // If `len < 8`, we duplicate bytes. We always put low memory at the end.
    // E.g., on little endian platforms:
    // `ABCD` will be read as `ABCDABCD`.
    // `ABCDE` will be read as `BCDEABCD`.
    // `ABCDEF` will be read as `CDEFABCD`.
    // `ABCDEFG` will be read as `DEFGABCD`.
    // `ABCDEFGH` will be read as `EFGHABCD`.
    // We also do not care about endianness. On big-endian platforms, bytes will
    // be permuted differently. We always shift low memory by 32, because that
    // can be pipelined earlier. Reading high memory requires computing
    // `p + len - 4`.
    let most_significant = (unaligned_load32(p.as_ptr()) as u64) << 32;
    let least_significant = unaligned_load32(p[p.len() - 4..].as_ptr()) as u64;
    most_significant | least_significant
}

// Reads 1 to 3 bytes from p. Some input bytes may be duplicated in output.
#[inline]
fn read1to3(p: &[u8]) -> u32 {
    // The trick used by this implementation is to avoid branches.
    // We always read three bytes by duplicating.
    // E.g.,
    // `A` is read as `AAA`.
    // `AB` is read as `ABB`.
    // `ABC` is read as `ABC`.
    // We always shift `p[0]` so that it can be pipelined better.
    // Other bytes require extra computation to find indices.
    let mem0 = ((p[0] as u32) << 16) | p[p.len() - 1] as u32;
    let mem1 = (p[p.len() / 2] as u32) << 8;
    mem0 | mem1
}

// Extremely weak mixture of length that is mixed into the state before
// combining the data. It is used only for small strings. This also ensures that
// we have high entropy in all bits of the state.
#[inline]
fn precombine_length_mix(state: u64, len: usize) -> u64 {
    debug_assert!(len + size_of::<u64>() <= size_of_val(&K_STATIC_RANDOM_DATA.0));
    let data = unsafe { unaligned_load64((K_STATIC_RANDOM_DATA.0.as_ptr() as *const u8).add(len)) };
    state ^ data
}

#[inline]
pub(crate) fn combine_raw(state: u64, value: u64) -> u64 {
    mix(state ^ value, K_MUL)
}

#[inline]
fn combine_small_contiguous(state: u64, first: &[u8]) -> u64 {
    let len = first.len();
    debug_assert!(len <= 8);

    let v = if len >= 4 {
        read4to8(first)
    } else if len > 0 {
        read1to3(first) as u64
    } else {
        0x57u64
    };
    combine_raw(state, v)
}

#[inline]
fn combine_contiguous_9to16(state: u64, first: &[u8]) -> u64 {
    let len = first.len();
    debug_assert!(len >= 9);
    debug_assert!(len <= 16);
    // Note: any time one half of the mix function becomes zero it will fail to
    // incorporate any bits from the other half. However, there is exactly 1 in
    // 2^64 values for each side that achieve this, and only when the size is
    // exactly 16 -- for smaller sizes there is an overlapping byte that makes
    // this impossible unless the seed is *also* incredibly unlucky.
    let (first, second) = read9to16(first);
    mix(state ^ first, K_MUL ^ second)
}

#[inline]
fn combine_contiguous_17to32(state: u64, first: &[u8]) -> u64 {
    let len = first.len();
    debug_assert!(len >= 17);
    debug_assert!(len <= 32);
    // Do two mixes of overlapping 16-byte ranges in parallel to minimize
    // latency.
    let m0 = mix(
        read8(first) ^ K_STATIC_RANDOM_DATA.0[1],
        read8(&first[8..]) ^ state,
    );

    let tail_16b = &first[len - 16..];
    let m1 = mix(
        read8(tail_16b) ^ K_STATIC_RANDOM_DATA.0[3],
        read8(&tail_16b[8..]) ^ state,
    );
    m0 ^ m1
}

#[cfg(all(target_feature = "sse2", target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "aes")]
fn low_level_hash33to64(seed: u64, ptr: &[u8]) -> u64 {
    unsafe {
        let len = ptr.len();
        debug_assert!(len > 32);
        debug_assert!(len <= 64);
        let state = _mm_set_epi64x(seed as i64, len as i64);
        let a = _mm_loadu_si128(ptr.as_ptr() as *const __m128i);
        let b = _mm_loadu_si128(ptr.as_ptr().add(16) as *const __m128i);
        let last32_ptr = ptr.as_ptr().add(len - 32);
        let c = _mm_loadu_si128(last32_ptr as *const __m128i);
        let d = _mm_loadu_si128(last32_ptr.add(16) as *const __m128i);

        // Bits of the second argument to _mm_aesdec_si128/_mm_aesenc_si128 are
        // XORed with the state argument after encryption.
        // We use each value as the first argument to shuffle all the bits around.
        // We do not add any salt to the state or loaded data, instead we vary
        // instructions used to mix bits _mm_aesdec_si128/_mm_aesenc_si128 and
        // _mm_add_epi64/_mm_sub_epi64.
        // _mm_add_epi64/_mm_sub_epi64 are combined to one instruction with data
        // loading like `vpaddq  xmm1, xmm0, xmmword ptr [rdi]`.
        let na = _mm_aesdec_si128(_mm_add_epi64(state, a), state);
        let nb = _mm_aesdec_si128(_mm_sub_epi64(state, b), state);
        let nc = _mm_aesenc_si128(_mm_add_epi64(state, c), state);
        let nd = _mm_aesenc_si128(_mm_sub_epi64(state, d), state);

        // We perform another round of encryption to mix bits between two halves of
        // the input.
        mix4x16vectors(na, nb, nc, nd)
    }
}

#[cfg(all(target_feature = "sse2", target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "aes")]
fn low_level_hash_len_gt64(seed: u64, data: &[u8]) -> u64 {
    let mut len = data.len();
    debug_assert!(len > 64);

    let mut ptr = data;
    let last_32 = &data[len - 32..];

    unsafe {
        // If we have more than 64 bytes, we're going to handle chunks of 64
        // bytes at a time. We're going to build up four separate hash states
        // which we will then hash together. This avoids short dependency chains.
        let mut state0 = _mm_set_epi64x(seed as i64, len as i64);
        let mut state1 = state0;
        let mut state2 = state1;
        let mut state3 = state2;

        // Mixing two 128-bit vectors at a time with corresponding states.
        // All variables are mixed slightly differently to avoid hash collision
        // due to trivial byte rotation.
        // We combine state and data with _mm_add_epi64/_mm_sub_epi64 before applying
        // AES encryption to make hash function dependent on the order of the blocks.
        // See comments in LowLevelHash33To64 for more considerations.
        let mut mix_ab = |p: &[u8]| {
            assert!(p.len() >= 16);

            // i128 a = *p;
            // i128 b = *(p + 16);
            // state0 = decrypt(state0 + a, state0);
            // state1 = decrypt(state1 - b, state1);
            let a = _mm_loadu_si128(p.as_ptr() as *const __m128i);
            let b = _mm_loadu_si128(p.as_ptr().add(16) as *const __m128i);
            state0 = _mm_aesdec_si128(_mm_add_epi64(state0, a), state0);
            state1 = _mm_aesdec_si128(_mm_sub_epi64(state1, b), state1);
        };

        let mut mix_cd = |p: &[u8]| {
            // i128 c = *p;
            // i128 d = *(p + 16);
            // state2 = encrypt(state2 + c, state2);
            // state3 = encrypt(state3 - d, state3);
            let c = _mm_loadu_si128(p.as_ptr() as *const __m128i);
            let d = _mm_loadu_si128(p.as_ptr().add(16) as *const __m128i);
            state2 = _mm_aesenc_si128(_mm_add_epi64(state2, c), state2);
            state3 = _mm_aesenc_si128(_mm_sub_epi64(state3, d), state3);
        };

        loop {
            prefetch_future_data_to_local_cache(ptr);
            mix_ab(ptr);
            mix_cd(&ptr[32..]);

            ptr = &ptr[64..];
            len -= 64;

            if len <= 64 {
                break;
            }
        }

        // We now have a data `ptr` with at most 64 bytes.
        if len > 32 {
            mix_ab(ptr);
        }
        mix_cd(last_32);

        mix4x16vectors(state0, state1, state2, state3)
    }
}

#[cfg(not(all(target_feature = "sse2", target_arch = "x86_64", not(miri))))]
fn low_level_hash33to64(seed: u64, ptr: &[u8]) -> u64 {
    let len = ptr.len();
    debug_assert!(len > 32);
    debug_assert!(len <= 64);

    let current_state = seed ^ K_STATIC_RANDOM_DATA.0[0] ^ len as u64;
    let last_32_ptr = &ptr[len - 32..];
    mix_32_bytes(last_32_ptr, mix_32_bytes(ptr, current_state))
}

#[cfg(not(all(target_feature = "sse2", target_arch = "x86_64", not(miri))))]
fn low_level_hash_len_gt64(seed: u64, data: &[u8]) -> u64 {
    let mut len = data.len();
    debug_assert!(len > 64);

    let mut ptr = data;
    let last_32 = &data[len - 32..];

    let mut current_state = seed ^ K_STATIC_RANDOM_DATA.0[0] ^ len as u64;
    // If we have more than 64 bytes, we're going to handle chunks of 64
    // bytes at a time. We're going to build up four separate hash states
    // which we will then hash together. This avoids short dependency chains.
    let mut duplicated_state0 = current_state;
    let mut duplicated_state1 = current_state;
    let mut duplicated_state2 = current_state;

    loop {
        prefetch_future_data_to_local_cache(ptr);

        let a = unaligned_load64(ptr.as_ptr());
        let b = unaligned_load64(ptr[8..].as_ptr());
        let c = unaligned_load64(ptr[16..].as_ptr());
        let d = unaligned_load64(ptr[24..].as_ptr());
        let e = unaligned_load64(ptr[32..].as_ptr());
        let f = unaligned_load64(ptr[40..].as_ptr());
        let g = unaligned_load64(ptr[48..].as_ptr());
        let h = unaligned_load64(ptr[56..].as_ptr());

        current_state = mix(a ^ K_STATIC_RANDOM_DATA.0[1], b ^ current_state);
        duplicated_state0 = mix(c ^ K_STATIC_RANDOM_DATA.0[2], d ^ duplicated_state0);

        duplicated_state1 = mix(e ^ K_STATIC_RANDOM_DATA.0[3], f ^ duplicated_state1);
        duplicated_state2 = mix(g ^ K_STATIC_RANDOM_DATA.0[4], h ^ duplicated_state2);

        ptr = &ptr[64..];
        len -= 64;

        if len <= 64 {
            break;
        }
    }

    current_state =
        (current_state ^ duplicated_state0) ^ (duplicated_state1.wrapping_add(duplicated_state2));
    // We now have a data `ptr` with at most 64 bytes and the current state
    // of the hashing state machine stored in current_state.
    if len > 32 {
        current_state = mix_32_bytes(ptr, current_state);
    }

    // We now have a data `ptr` with at most 32 bytes and the current state
    // of the hashing state machine stored in current_state. But we can
    // safely read from `ptr + len - 32`.
    mix_32_bytes(last_32, current_state)
}

fn low_level_hash_len_gt32(seed: u64, data: &[u8]) -> u64 {
    let len = data.len();
    debug_assert!(len > 32);

    if unlikely(len > 64) {
        return unsafe { low_level_hash_len_gt64(seed, data) };
    }
    unsafe { low_level_hash33to64(seed, data) }
}

#[inline]
fn hash_block_on32bit(state: u64, data: &[u8]) -> u64 {
    // TODO(b/417141985): expose and use CityHash32WithSeed.
    let len = data.len();
    combine_raw(state + len as u64, city::hash32(data) as u64)
}

#[inline]
fn hash_block_on64bit(state: u64, data: &[u8]) -> u64 {
    low_level_hash_len_gt32(state, data)
}

fn split_and_combine_on32bit(mut state: u64, first: &[u8]) -> u64 {
    let mut len = first.len();
    let mut ptr = first;
    while len >= PIECEWISE_CHUNK_SIZE {
        state = hash_block_on32bit(state, &ptr[0..PIECEWISE_CHUNK_SIZE]);
        len -= PIECEWISE_CHUNK_SIZE;
        ptr = &ptr[PIECEWISE_CHUNK_SIZE..];
    }
    // Do not call combine_contiguous_on32bit for empty range since it is modifying
    // state.
    if len == 0 {
        return state;
    }
    // Handle the remainder.
    combine_contiguous_on32bit(state, ptr)
}

fn split_and_combine_on64bit(mut state: u64, first: &[u8]) -> u64 {
    let mut len = first.len();
    let mut ptr = first;
    while len >= PIECEWISE_CHUNK_SIZE {
        state = hash_block_on64bit(state, &ptr[0..PIECEWISE_CHUNK_SIZE]);
        len -= PIECEWISE_CHUNK_SIZE;
        ptr = &ptr[PIECEWISE_CHUNK_SIZE..];
    }
    // Do not call combine_contiguous_on64bit for empty range since it is modifying
    // state.
    if len == 0 {
        return state;
    }
    // Handle the remainder.
    unsafe { combine_contiguous_on64bit(state, ptr) }
}

fn combine_large_contiguous_on32bit_length_gt8(state: u64, first: &[u8]) -> u64 {
    let len = first.len();
    debug_assert!(len > 8);
    debug_assert_eq!(size_of::<usize>(), 4);
    if likely(len <= PIECEWISE_CHUNK_SIZE) {
        return hash_block_on32bit(state, first);
    }
    split_and_combine_on32bit(state, first)
}

fn combine_large_contiguous_on64bit_length_gt32(state: u64, first: &[u8]) -> u64 {
    let len = first.len();
    debug_assert!(len > 32);
    debug_assert_eq!(size_of::<usize>(), 8);
    if likely(len <= PIECEWISE_CHUNK_SIZE) {
        return hash_block_on64bit(state, first);
    }
    split_and_combine_on64bit(state, first)
}

#[cfg(test)]
mod tests {
    use crate::internal::low_level::combine_large_contiguous_on64bit_length_gt32;

    use base64::engine::general_purpose;
    use base64::Engine;

    const K_NUM_GOLDEN_OUTPUTS: usize = 95;

    static CASES: [(&str, u64); 95] = [
        (
            "VprUGNH+5NnNRaORxgH/ySrZFQFDL+4VAodhfBNinmn8cg==",
            0x531858a40bfa7ea1,
        ),
        (
            "gc1xZaY+q0nPcUvOOnWnT3bqfmT/geth/f7Dm2e/DemMfk4=",
            0x86689478a7a7e8fa,
        ),
        (
            "Mr35fIxqx1ukPAL0su1yFuzzAU3wABCLZ8+ZUFsXn47UmAph",
            0x4ec948b8e7f27288,
        ),
        (
            "A9G8pw2+m7+rDtWYAdbl8tb2fT7FFo4hLi2vAsa5Y8mKH3CX3g==",
            0xce46c7213c10032,
        ),
        (
            "DFaJGishGwEHDdj9ixbCoaTjz9KS0phLNWHVVdFsM93CvPft3hM=",
            0xf63e96ee6f32a8b6,
        ),
        (
            "7+Ugx+Kr3aRNgYgcUxru62YkTDt5Hqis+2po81hGBkcrJg4N0uuy",
            0x1cfe85e65fc5225,
        ),
        (
            "H2w6O8BUKqu6Tvj2xxaecxEI2wRgIgqnTTG1WwOgDSINR13Nm4d4Vg==",
            0x45c474f1cee1d2e8,
        ),
        (
            "1XBMnIbqD5jy65xTDaf6WtiwtdtQwv1dCVoqpeKj+7cTR1SaMWMyI04=",
            0x6e024e14015f329c,
        ),
        (
            "znZbdXG2TSFrKHEuJc83gPncYpzXGbAebUpP0XxzH0rpe8BaMQ17nDbt",
            0x760c40502103ae1c,
        ),
        (
            "ylu8Atu13j1StlcC1MRMJJXIl7USgDDS22HgVv0WQ8hx/8pNtaiKB17hCQ==",
            0x17fd05c3c560c320,
        ),
        (
            "M6ZVVzsd7vAvbiACSYHioH/440dp4xG2mLlBnxgiqEvI/aIEGpD0Sf4VS0g=",
            0x8b34200a6f8e90d9,
        ),
        (
            "li3oFSXLXI+ubUVGJ4blP6mNinGKLHWkvGruun85AhVn6iuMtocbZPVhqxzn",
            0x6be89e50818bdf69,
        ),
        (
            "kFuQHuUCqBF3Tc3hO4dgdIp223ShaCoog48d5Do5zMqUXOh5XpGK1t5XtxnfGA==",
            0xfb389773315b47d8,
        ),
        (
            "jWmOad0v0QhXVJd1OdGuBZtDYYS8wBVHlvOeTQx9ZZnm8wLEItPMeihj72E0nWY=",
            0x4f2512a23f61efee,
        ),
        (
            "z+DHU52HaOQdW4JrZwDQAebEA6rm13Zg/9lPYA3txt3NjTBqFZlOMvTRnVzRbl23",
            0x59ccd92fc16c6fda,
        ),
        (
            "MmBiGDfYeTayyJa/tVycg+rN7f9mPDFaDc+23j0TlW9094er0ADigsl4QX7V3gG/qw==",
            0x25c5a7f5bd330919,
        ),
        (
            "774RK+9rOL4iFvs1q2qpo/JVc/I39buvNjqEFDtDvyoB0FXxPI2vXqOrk08VPfIHkmU=",
            0x51df4174d34c97d7,
        ),
        (
            "+slatXiQ7/2lK0BkVUI1qzNxOOLP3I1iK6OfHaoxgqT63FpzbElwEXSwdsryq3UlHK0I",
            0x80ce6d76f89cb57,
        ),
        (
            "64mVTbQ47dHjHlOHGS/hjJwr/K2frCNpn87exOqMzNUVYiPKmhCbfS7vBUce5tO6Ec9osQ==",
            0x20961c911965f684,
        ),
        (
            "fIsaG1r530SFrBqaDj1kqE0AJnvvK8MNEZbII2Yw1OK77v0V59xabIh0B5axaz/+a2V5WpA=",
            0x4e5b926ec83868e7,
        ),
        (
            "PGih0zDEOWCYGxuHGDFu9Ivbff/iE7BNUq65tycTR2R76TerrXALRosnzaNYO5fjFhTi+CiS",
            0x3927b30b922eecef,
        ),
        (
            "RnpA/zJnEnnLjmICORByRVb9bCOgxF44p3VMiW10G7PvW7IhwsWajlP9kIwNA9FjAD2GoQHk2Q==",
            0xbd0291284a49b61c,
        ),
        (
            "qFklMceaTHqJpy2qavJE+EVBiNFOi6OxjOA3LeIcBop1K7w8xQi3TrDk+BrWPRIbfprszSaPfrI=",
            0x73a77c575bcc956,
        ),
        (
            "cLbfUtLl3EcQmITWoTskUR8da/VafRDYF/ylPYwk7/zazk6ssyrzxMN3mmSyvrXR2yDGNZ3WDrTT",
            0x766a0e2ade6d09a6,
        ),
        (
            "s/Jf1+FbsbCpXWPTUSeWyMH6e4CvTFvPE5Fs6Z8hvFITGyr0dtukHzkI84oviVLxhM1xMxrMAy1dbw==",
            0x2599f4f905115869,
        ),
        (
            "FvyQ00+j7nmYZVQ8hI1Edxd0AWplhTfWuFGiu34AK5X8u2hLX1bE97sZM0CmeLe+7LgoUT1fJ/axybE=",
            0xd8256e5444d21e53,
        ),
        (
            "L8ncxMaYLBH3g9buPu8hfpWZNlOF7nvWLNv9IozH07uQsIBWSKxoPy8+LW4tTuzC6CIWbRGRRD1sQV/4",
            0xf664a91333fb8dfd,
        ),
        (
            "CDK0meI07yrgV2kQlZZ+wuVqhc2NmzqeLH7bmcA6kchsRWFPeVF5Wqjjaj556ABeUoUr3yBmfU3kWOakkg==",
            0x9625b859be372cd1,
        ),
        (
            "d23/vc5ONh/HkMiq+gYk4gaCNYyuFKwUkvn46t+dfVcKfBTYykr4kdvAPNXGYLjM4u1YkAEFpJP+nX7eOvs=",
            0x7b99940782e29898,
        ),
        (
            "NUR3SRxBkxTSbtQORJpu/GdR6b/h6sSGfsMj/KFd99ahbh+9r7LSgSGmkGVB/mGoT0pnMTQst7Lv2q6QN6Vm",
            0x4fe12fa5383b51a8,
        ),
        (
            "2BOFlcI3Z0RYDtS9T9Ie9yJoXlOdigpPeeT+CRujb/O39Ih5LPC9hP6RQk1kYESGyaLZZi3jtabHs7DiVx/VDg==",
            0xe2ccb09ac0f5b4b6,
        ),
        (
            "FF2HQE1FxEvWBpg6Z9zAMH+Zlqx8S1JD/wIlViL6ZDZY63alMDrxB0GJQahmAtjlm26RGLnjW7jmgQ4Ie3I+014=",
            0x7d0a37adbd7b753b,
        ),
        (
            "tHmO7mqVL/PX11nZrz50Hc+M17Poj5lpnqHkEN+4bpMx/YGbkrGOaYjoQjgmt1X2QyypK7xClFrjeWrCMdlVYtbW",
            0xd3ae96ef9f7185f2,
        ),
        (
            "/WiHi9IQcxRImsudkA/KOTqGe8/gXkhKIHkjddv5S9hi02M049dIK3EUyAEjkjpdGLUs+BN0QzPtZqjIYPOgwsYE9g==",
            0x4fb88ea63f79a0d8,
        ),
        (
            "qds+1ExSnU11L4fTSDz/QE90g4Jh6ioqSh3KDOTOAo2pQGL1k/9CCC7J23YF27dUTzrWsCQA2m4epXoCc3yPHb3xElA=",
            0xed564e259bb5ebe9,
        ),
        (
            "8FVYHx40lSQPTHheh08Oq0/pGm2OlG8BEf8ezvAxHuGGdgCkqpXIueJBF2mQJhTfDy5NncO8ntS7vaKs7sCNdDaNGOEi",
            0x3e3256b60c428000,
        ),
        (
            "4ZoEIrJtstiCkeew3oRzmyJHVt/pAs2pj0HgHFrBPztbQ10NsQ/lM6DM439QVxpznnBSiHMgMQJhER+70l72LqFTO1JiIQ==",
            0xfb05bad59ec8705,
        ),
        (
            "hQPtaYI+wJyxXgwD5n8jGIKFKaFA/P83KqCKZfPthnjwdOFysqEOYwAaZuaaiv4cDyi9TyS8hk5cEbNP/jrI7q6pYGBLbsM=",
            0xafdc251dbf97b5f8,
        ),
        (
            "S4gpMSKzMD7CWPsSfLeYyhSpfWOntyuVZdX1xSBjiGvsspwOZcxNKCRIOqAA0moUfOh3I5+juQV4rsqYElMD/gWfDGpsWZKQ",
            0x10ec9c92ddb5dcbc,
        ),
        (
            "oswxop+bthuDLT4j0PcoSKby4LhF47ZKg8K17xxHf74UsGCzTBbOz0MM8hQEGlyqDT1iUiAYnaPaUpL2mRK0rcIUYA4qLt5uOw==",
            0x9a767d5822c7dac4,
        ),
        (
            "0II/697p+BtLSjxj5989OXI004TogEb94VUnDzOVSgMXie72cuYRvTFNIBgtXlKfkiUjeqVpd4a+n5bxNOD1TGrjQtzKU5r7obo=",
            0xee46254080d6e2db,
        ),
        (
            "E84YZW2qipAlMPmctrg7TKlwLZ68l4L+c0xRDUfyyFrA4MAti0q9sHq3TDFviH0Y+Kq3tEE5srWFA8LM9oomtmvm5PYxoaarWPLc",
            0xbbb669588d8bf398,
        ),
        (
            "x3pa4HIElyZG0Nj7Vdy9IdJIR4izLmypXw5PCmZB5y68QQ4uRaVVi3UthsoJROvbjDJkP2DQ6L/eN8pFeLFzNPKBYzcmuMOb5Ull7w==",
            0xdc2afaa529beef44,
        ),
        (
            "jVDKGYIuWOP/QKLdd2wi8B2VJA8Wh0c8PwrXJVM8FOGM3voPDVPyDJOU6QsBDPseoR8uuKd19OZ/zAvSCB+zlf6upAsBlheUKgCfKww=",
            0xf1f67391d45013a8,
        ),
        (
            "mkquunhmYe1aR2wmUz4vcvLEcKBoe6H+kjUok9VUn2+eTSkWs4oDDtJvNCWtY5efJwg/j4PgjRYWtqnrCkhaqJaEvkkOwVfgMIwF3e+d",
            0x16fce2b8c65a3429,
        ),
        (
            "fRelvKYonTQ+s+rnnvQw+JzGfFoPixtna0vzcSjiDqX5s2Kg2//UGrK+AVCyMUhO98WoB1DDbrsOYSw2QzrcPe0+3ck9sePvb+Q/IRaHbw==",
            0xf4b096699f49fe67,
        ),
        (
            "DUwXFJzagljo44QeJ7/6ZKw4QXV18lhkYT2jglMr8WB3CHUU4vdsytvw6AKv42ZcG6fRkZkq9fpnmXy6xG0aO3WPT1eHuyFirAlkW+zKtwg=",
            0xca584c4bc8198682,
        ),
        (
            "cYmZCrOOBBongNTr7e4nYn52uQUy2mfe48s50JXx2AZ6cRAt/xRHJ5QbEoEJOeOHsJyM4nbzwFm++SlT6gFZZHJpkXJ92JkR86uS/eV1hJUR",
            0xed269fc3818b6aad,
        ),
        (
            "EXeHBDfhwzAKFhsMcH9+2RHwV+mJaN01+9oacF6vgm8mCXRd6jeN9U2oAb0of5c5cO4i+Vb/LlHZSMI490SnHU0bejhSCC2gsC5d2K30ER3iNA==",
            0x33f253cbb8fe66a8,
        ),
        (
            "FzkzRYoNjkxFhZDso94IHRZaJUP61nFYrh5MwDwv9FNoJ5jyNCY/eazPZk+tbmzDyJIGw2h3GxaWZ9bSlsol/vK98SbkMKCQ/wbfrXRLcDzdd/8=",
            0xd0b76b2c1523d99c,
        ),
        (
            "Re4aXISCMlYY/XsX7zkIFR04ta03u4zkL9dVbLXMa/q6hlY/CImVIIYRN3VKP4pnd0AUr/ugkyt36JcstAInb4h9rpAGQ7GMVOgBniiMBZ/MGU7H",
            0xfd28f0811a2a237f,
        ),
        (
            "ueLyMcqJXX+MhO4UApylCN9WlTQ+ltJmItgG7vFUtqs2qNwBMjmAvr5u0sAKd8jpzV0dDPTwchbIeAW5zbtkA2NABJV6hFM48ib4/J3A5mseA3cS8w==",
            0x6261fb136482e84,
        ),
        (
            "6Si7Yi11L+jZMkwaN+GUuzXMrlvEqviEkGOilNq0h8TdQyYKuFXzkYc/q74gP3pVCyiwz9KpVGMM9vfnq36riMHRknkmhQutxLZs5fbmOgEO69HglCU=",
            0x458efc750bca7c3a,
        ),
        (
            "Q6AbOofGuTJOegPh9Clm/9crtUMQqylKrTc1fhfJo1tqvpXxhU4k08kntL1RG7woRnFrVh2UoMrL1kjin+s9CanT+y4hHwLqRranl9FjvxfVKm3yvg68",
            0xa7e69ff84e5e7c27,
        ),
        (
            "ieQEbIPvqY2YfIjHnqfJiO1/MIVRk0RoaG/WWi3kFrfIGiNLCczYoklgaecHMm/1sZ96AjO+a5stQfZbJQwS7Sc1ODABEdJKcTsxeW2hbh9A6CFzpowP1A==",
            0x3c59bfd0c29efe9e,
        ),
        (
            "zQUv8hFB3zh2GGl3KTvCmnfzE+SUgQPVaSVIELFX5H9cE3FuVFGmymkPQZJLAyzC90Cmi8GqYCvPqTuAAB//XTJxy4bCcVArgZG9zJXpjowpNBfr3ngWrSE=",
            0x10befacc6afd298d,
        ),
        (
            "US4hcC1+op5JKGC7eIs8CUgInjKWKlvKQkapulxW262E/B2ye79QxOexf188u2mFwwe3WTISJHRZzS61IwljqAWAWoBAqkUnW8SHmIDwHUP31J0p5sGdP47L",
            0x41d5320b0a38efa7,
        ),
        (
            "9bHUWFna2LNaGF6fQLlkx1Hkt24nrkLE2CmFdWgTQV3FFbUe747SSqYw6ebpTa07MWSpWRPsHesVo2B9tqHbe7eQmqYebPDFnNqrhSdZwFm9arLQVs+7a3Ic6A==",
            0x58db1c7450fe17f3,
        ),
        (
            "Kb3DpHRUPhtyqgs3RuXjzA08jGb59hjKTOeFt1qhoINfYyfTt2buKhD6YVffRCPsgK9SeqZqRPJSyaqsa0ovyq1WnWW8jI/NhvAkZTVHUrX2pC+cD3OPYT05Dag=",
            0x6098c055a335b7a6,
        ),
        (
            "gzxyMJIPlU+bJBwhFUCHSofZ/319LxqMoqnt3+L6h2U2+ZXJCSsYpE80xmR0Ta77Jq54o92SMH87HV8dGOaCTuAYF+lDL42SY1P316Cl0sZTS2ow3ZqwGbcPNs/1",
            0x1bbacec67845a801,
        ),
        (
            "uR7V0TW+FGVMpsifnaBAQ3IGlr1wx5sKd7TChuqRe6OvUXTlD4hKWy8S+8yyOw8lQabism19vOQxfmocEOW/vzY0pEa87qHrAZy4s9fH2Bltu8vaOIe+agYohhYORQ==",
            0xc419cfc7442190,
        ),
        (
            "1UR5eoo2aCwhacjZHaCh9bkOsITp6QunUxHQ2SfeHv0imHetzt/Z70mhyWZBalv6eAx+YfWKCUib2SHDtz/A2dc3hqUWX5VfAV7FQsghPUAtu6IiRatq4YSLpDvKZBQ=",
            0xc95e510d94ba270c,
        ),
        (
            "opubR7H63BH7OtY+Avd7QyQ25UZ8kLBdFDsBTwZlY6gA/u+x+czC9AaZMgmQrUy15DH7YMGsvdXnviTtI4eVI4aF1H9Rl3NXMKZgwFOsdTfdcZeeHVRzBBKX8jUfh1il",
            0xff1ae05c98089c3f,
        ),
        (
            "DC0kXcSXtfQ9FbSRwirIn5tgPri0sbzHSa78aDZVDUKCMaBGyFU6BmrulywYX8yzvwprdLsoOwTWN2wMjHlPDqrvVHNEjnmufRDblW+nSS+xtKNs3N5xsxXdv6JXDrAB/Q==",
            0x90c02b8dceced493,
        ),
        (
            "BXRBk+3wEP3Lpm1y75wjoz+PgB0AMzLe8tQ1AYU2/oqrQB2YMC6W+9QDbcOfkGbeH+b7IBkt/gwCMw2HaQsRFEsurXtcQ3YwRuPz5XNaw5NAvrNa67Fm7eRzdE1+hWLKtA8=",
            0x9f8a76697ab1aa36,
        ),
        (
            "RRBSvEGYnzR9E45Aps/+WSnpCo/X7gJLO4DRnUqFrJCV/kzWlusLE/6ZU6RoUf2ROwcgEvUiXTGjLs7ts3t9SXnJHxC1KiOzxHdYLMhVvgNd3hVSAXODpKFSkVXND55G2L1W",
            0x6ba1bf3d811a531d,
        ),
        (
            "jeh6Qazxmdi57pa9S3XSnnZFIRrnc6s8QLrah5OX3SB/V2ErSPoEAumavzQPkdKF1/SfvmdL+qgF1C+Yawy562QaFqwVGq7+tW0yxP8FStb56ZRgNI4IOmI30s1Ei7iops9Uuw==",
            0x6a418974109c67b4,
        ),
        (
            "6QO5nnDrY2/wrUXpltlKy2dSBcmK15fOY092CR7KxAjNfaY+aAmtWbbzQk3MjBg03x39afSUN1fkrWACdyQKRaGxgwq6MGNxI6W+8DLWJBHzIXrntrE/ml6fnNXEpxplWJ1vEs4=",
            0x8472f1c2b3d230a3,
        ),
        (
            "0oPxeEHhqhcFuwonNfLd5jF3RNATGZS6NPoS0WklnzyokbTqcl4BeBkMn07+fDQv83j/BpGUwcWO05f3+DYzocfnizpFjLJemFGsls3gxcBYxcbqWYev51tG3lN9EvRE+X9+Pwww",
            0x5e06068f884e73a7,
        ),
        (
            "naSBSjtOKgAOg8XVbR5cHAW3Y+QL4Pb/JO9/oy6L08wvVRZqo0BrssMwhzBP401Um7A4ppAupbQeJFdMrysY34AuSSNvtNUy5VxjNECwiNtgwYHw7yakDUv8WvonctmnoSPKENegQg==",
            0x55290b1a8f170f59,
        ),
        (
            "vPyl8DxVeRe1OpilKb9KNwpGkQRtA94UpAHetNh+95V7nIW38v7PpzhnTWIml5kw3So1Si0TXtIUPIbsu32BNhoH7QwFvLM+JACgSpc5e3RjsL6Qwxxi11npwxRmRUqATDeMUfRAjxg=",
            0x5501cfd83dfe706a,
        ),
        (
            "QC9i2GjdTMuNC1xQJ74ngKfrlA4w3o58FhvNCltdIpuMhHP1YsDA78scQPLbZ3OCUgeQguYf/vw6zAaVKSgwtaykqg5ka/4vhz4hYqWU5ficdXqClHl+zkWEY26slCNYOM5nnDlly8Cj",
            0xe43ed13d13a66990,
        ),
        (
            "7CNIgQhAHX27nxI0HeB5oUTnTdgKpRDYDKwRcXfSFGP1XeT9nQF6WKCMjL1tBV6x7KuJ91GZz11F4c+8s+MfqEAEpd4FHzamrMNjGcjCyrVtU6y+7HscMVzr7Q/ODLcPEFztFnwjvCjmHw==",
            0xdf43bc375cf5283f,
        ),
        (
            "Qa/hC2RPXhANSospe+gUaPfjdK/yhQvfm4cCV6/pdvCYWPv8p1kMtKOX3h5/8oZ31fsmx4Axphu5qXJokuhZKkBUJueuMpxRyXpwSWz2wELx5glxF7CM0Fn+OevnkhUn5jsPlG2r5jYlVn8=",
            0x8112b806d288d7b5,
        ),
        (
            "kUw/0z4l3a89jTwN5jpG0SHY5km/IVhTjgM5xCiPRLncg40aqWrJ5vcF891AOq5hEpSq0bUCJUMFXgct7kvnys905HjerV7Vs1Gy84tgVJ70/2+pAZTsB/PzNOE/G6sOj4+GbTzkQu819OLB",
            0xd52a18abb001cb46,
        ),
        (
            "VDdfSDbO8Tdj3T5W0XM3EI7iHh5xpIutiM6dvcJ/fhe23V/srFEkDy5iZf/VnA9kfi2C79ENnFnbOReeuZW1b3MUXB9lgC6U4pOTuC+jHK3Qnpyiqzj7h3ISJSuo2pob7vY6VHZo6Fn7exEqHg==",
            0xe12b76a2433a1236,
        ),
        (
            "Ldfvy3ORdquM/R2fIkhH/ONi69mcP1AEJ6n/oropwecAsLJzQSgezSY8bEiEs0VnFTBBsW+RtZY6tDj03fnb3amNUOq1b7jbqyQkL9hpl+2Z2J8IaVSeownWl+bQcsR5/xRktIMckC5AtF4YHfU=",
            0x175bf7319cf1fa00,
        ),
        (
            "BrbNpb42+VzZAjJw6QLirXzhweCVRfwlczzZ0VX2xluskwBqyfnGovz5EuX79JJ31VNXa5hTkAyQat3lYKRADTdAdwE5PqM1N7YaMqqsqoAAAeuYVXuk5eWCykYmClNdSspegwgCuT+403JigBzi",
            0xd63d57b3f67525ae,
        ),
        (
            "gB3NGHJJvVcuPyF0ZSvHwnWSIfmaI7La24VMPQVoIIWF7Z74NltPZZpx2f+cocESM+ILzQW9p+BC8x5IWz7N4Str2WLGKMdgmaBfNkEhSHQDU0IJEOnpUt0HmjhFaBlx0/LTmhua+rQ6Wup8ezLwfg==",
            0x933faea858832b73,
        ),
        (
            "hTKHlRxx6Pl4gjG+6ksvvj0CWFicUg3WrPdSJypDpq91LUWRni2KF6+81ZoHBFhEBrCdogKqeK+hy9bLDnx7g6rAFUjtn1+cWzQ2YjiOpz4+ROBB7lnwjyTGWzJD1rXtlso1g2qVH8XJVigC5M9AIxM=",
            0x53d061e5f8e7c04f,
        ),
        (
            "IWQBelSQnhrr0F3BhUpXUIDauhX6f95Qp+A0diFXiUK7irwPG1oqBiqHyK/SH/9S+rln9DlFROAmeFdH0OCJi2tFm4afxYzJTFR4HnR4cG4x12JqHaZLQx6iiu6CE3rtWBVz99oAwCZUOEXIsLU24o2Y",
            0xdb4124556dd515e0,
        ),
        (
            "TKo+l+1dOXdLvIrFqeLaHdm0HZnbcdEgOoLVcGRiCbAMR0j5pIFw8D36tefckAS1RCFOH5IgP8yiFT0Gd0a2hI3+fTKA7iK96NekxWeoeqzJyctc6QsoiyBlkZerRxs5RplrxoeNg29kKDTM0K94mnhD9g==",
            0x4fb31a0dd681ee71,
        ),
        (
            "YU4e7G6EfQYvxCFoCrrT0EFgVLHFfOWRTJQJ5gxM3G2b+1kJf9YPrpsxF6Xr6nYtS8reEEbDoZJYqnlk9lXSkVArm88Cqn6d25VCx3+49MqC0trIlXtb7SXUUhwpJK16T0hJUfPH7s5cMZXc6YmmbFuBNPE=",
            0x27cc72eefa138e4c,
        ),
        (
            "/I/eImMwPo1U6wekNFD1Jxjk9XQVi1D+FPdqcHifYXQuP5aScNQfxMAmaPR2XhuOQhADV5tTVbBKwCDCX4E3jcDNHzCiPvViZF1W27txaf2BbFQdwKrNCmrtzcluBFYu0XZfc7RU1RmxK/RtnF1qHsq/O4pp",
            0x44bc2dfba4bd3ced,
        ),
        (
            "CJTT9WGcY2XykTdo8KodRIA29qsqY0iHzWZRjKHb9alwyJ7RZAE3V5Juv4MY3MeYEr1EPCCMxO7yFXqT8XA8YTjaMp3bafRt17Pw8JC4iKJ1zN+WWKOESrj+3aluGQqn8z1EzqY4PH7rLG575PYeWsP98BugdA==",
            0x242da1e3a439bed8,
        ),
        (
            "ZlhyQwLhXQyIUEnMH/AEW27vh9xrbNKJxpWGtrEmKhd+nFqAfbeNBQjW0SfG1YI0xQkQMHXjuTt4P/EpZRtA47ibZDVS8TtaxwyBjuIDwqcN09eCtpC+Ls+vWDTLmBeDM3u4hmzz4DQAYsLiZYSJcldg9Q3wszw=",
            0xdc559c746e35c139,
        ),
        (
            "v2KU8y0sCrBghmnm8lzGJlwo6D6ObccAxCf10heoDtYLosk4ztTpLlpSFEyu23MLA1tJkcgRko04h19QMG0mOw/wc93EXAweriBqXfvdaP85sZABwiKO+6rtS9pacRVpYYhHJeVTQ5NzrvBvi1huxAr+xswhVMfL",
            0xd0b0350275b9989,
        ),
        (
            "QhKlnIS6BuVCTQsnoE67E/yrgogE8EwO7xLaEGei26m0gEU4OksefJgppDh3X0x0Cs78Dr9IHK5b977CmZlrTRmwhlP8pM+UzXPNRNIZuN3ntOum/QhUWP8SGpirheXENWsXMQ/nxtxakyEtrNkKk471Oov9juP8oQ==",
            0xb04489e41d17730c,
        ),
        (
            "/ZRMgnoRt+Uo6fUPr9FqQvKX7syhgVqWu+WUSsiQ68UlN0efSP6Eced5gJZL6tg9gcYJIkhjuQNITU0Q3TjVAnAcobgbJikCn6qZ6pRxKBY4MTiAlfGD3T7R7hwJwx554MAy++Zb/YUFlnCaCJiwQMnowF7aQzwYFCo=",
            0x2217285eb4572156,
        ),
        (
            "NB7tU5fNE8nI+SXGfipc7sRkhnSkUF1krjeo6k+8FITaAtdyz+o7mONgXmGLulBPH9bEwyYhKNVY0L+njNQrZ9YC2aXsFD3PdZsxAFaBT3VXEzh+NGBTjDASNL3mXyS8Yv1iThGfHoY7T4aR0NYGJ+k+pR6f+KrPC96M",
            0x12c2e8e68aede73b,
        ),
        (
            "8T6wrqCtEO6/rwxF6lvMeyuigVOLwPipX/FULvwyu+1wa5sQGav/2FsLHUVn6cGSi0LlFwLewGHPFJDLR0u4t7ZUyM//x6da0sWgOa5hzDqjsVGmjxEHXiaXKW3i4iSZNuxoNbMQkIbVML+DkYu9ND0O2swg4itGeVSzXA==",
            0x4d612125bdc4fd00,
        ),
        (
            "Ntf1bMRdondtMv1CYr3G80iDJ4WSAlKy5H34XdGruQiCrnRGDBa+eUi7vKp4gp3BBcVGl8eYSasVQQjn7MLvb3BjtXx6c/bCL7JtpzQKaDnPr9GWRxpBXVxKREgMM7d8lm35EODv0w+hQLfVSh8OGs7fsBb68nNWPLeeSOo=",
            0x81826b553954464e,
        ),
        (
            "VsSAw72Ro6xks02kaiLuiTEIWBC5bgqr4WDnmP8vglXzAhixk7td926rm9jNimL+kroPSygZ9gl63aF5DCPOACXmsbmhDrAQuUzoh9ZKhWgElLQsrqo1KIjWoZT5b5QfVUXY9lSIBg3U75SqORoTPq7HalxxoIT5diWOcJQi",
            0xc2e5d345dc0ddd2d,
        ),
        (
            "j+loZ+C87+bJxNVebg94gU0mSLeDulcHs84tQT7BZM2rzDSLiCNxUedHr1ZWJ9ejTiBa0dqy2I2ABc++xzOLcv+//YfibtjKtYggC6/3rv0XCc7xu6d/O6xO+XOBhOWAQ+IHJVHf7wZnDxIXB8AUHsnjEISKj7823biqXjyP3g==",
            0x3da6830a9e32631e,
        ),
        (
            "f3LlpcPElMkspNtDq5xXyWU62erEaKn7RWKlo540gR6mZsNpK1czV/sOmqaq8XAQLEn68LKj6/cFkJukxRzCa4OF1a7cCAXYFp9+wZDu0bw4y63qbpjhdCl8GO6Z2lkcXy7KOzbPE01ukg7+gN+7uKpoohgAhIwpAKQXmX5xtd0=",
            0xc9ae5c8759b4877a,
        ),
    ];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.2,aes")]
    unsafe fn get_k_golden() -> [u64; K_NUM_GOLDEN_OUTPUTS] {
        [
            0xd6bdb2c9ba5e55f2,
            0xffd3e23d4115a8ae,
            0x2c3218ef486127de,
            0x554fa7f3a262b886,
            0x06304cbf82e312d3,
            0x490b3fb5af80622c,
            0x7398a90b8cc59c5d,
            0x65fb3168b98030ab,
            0xd4564363c53617bb,
            0x0545c26351925fe7,
            0xc30700723b634bf4,
            0xfb23a140a76dbe94,
            0x2fa1467fe218a47c,
            0x92e05ec3a7b966eb,
            0x6112b56e5624dd50,
            0x8760801365f9d722,
            0x41f7187b61db0e5e,
            0x7fe9188a1f5f50ad,
            0x25800bd4c2002ef1,
            0x91fecd33a78ef0aa,
            0x93986ad71e983613,
            0xe4c78173c7ea537b,
            0x0bbdc2bcabdb50b1,
            0xd9aa134df2d87623,
            0x6c4907c9477a9409,
            0xc3e418a5dbda52e5,
            0x4d24f3e9d0dda93a,
            0xcdb565a363dbe45f,
            0xa95f228c8ee57478,
            0x6b8f00bab5130227,
            0x2d05a0f44818b67a,
            0xd6bf7d990b5f44cb,
            0xa3608bdb4712861a,
            0xf20c33e5e355330b,
            0xbc86e1b13130180d,
            0x0848221b397b839a,
            0x17cc0acf44a7e210,
            0xc18c6dc584fe0f62,
            0x896c7858a59f991d,
            0xeab1e6d7d2856ed7,
            0x7e4b2d99c23edc51,
            0x9aeeeb7fa46e7cf0,
            0x161b9f2e3611790f,
            0x5f82aae18d971b36,
            0x8d0dd9965881e162,
            0x56700ea26285895a,
            0xcd919c86c29a053e,
            0x3e5d589282d9a722,
            0x92caee9f48a66604,
            0x7e1a2fd9b06f14b0,
            0xce1d5293f95b0178,
            0x8101361290e70a11,
            0x570e3e9c9eafc1c6,
            0x77b6241926a7a568,
            0x313e5cb34f346699,
            0xab8ebeab0514b82b,
            0x6e0a43763a310408,
            0x761b76ec22b2e440,
            0x4238c84a9ec00528,
            0xb9ea1f6d4d5552af,
            0xd21f8f110b9dc060,
            0xb3d3842b69ac3689,
            0xd0a88aa1dcf59869,
            0xf3f69f637b123403,
            0xf5f34b1068cac7da,
            0xe69a08d604774abf,
            0x57648d3a73332437,
            0x9762947f5013d00d,
            0x35c5d734a0015922,
            0xbee2fe5a104ce209,
            0xedb060efa6efca34,
            0x5ccf0f4786d97bc2,
            0x1ef8ed72e80d7bef,
            0x58522deb49c5e30f,
            0xde97cd2a6f8bd13b,
            0x3fae37c6f9855d09,
            0xea99ae786feca261,
            0x8c6d1d46670b0943,
            0x84658b2a232c7bfb,
            0x7058b7a7968de394,
            0x0d44fba68e25aa8f,
            0xc7f687020f8eb00b,
            0xbf9671e1196153d6,
            0x1009be891b7f83e7,
            0x4f9457fb4aa12865,
            0x30a49d9563643b32,
            0x0302e2c5b46d5a3a,
            0x77553f42fb0bfbf7,
            0x26b95e89f0077110,
            0x76ce68ebe01191ba,
            0x724110fb509e4376,
            0xebe74b016b5cfb88,
            0x3b0fe11dcf175fc9,
            0x20b737b9c0490538,
            0x0db21c429b45fd17,
        ]
    }

    #[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"),)))]
    fn get_k_golden() -> [u64; K_NUM_GOLDEN_OUTPUTS] {
        [
            0x669da02f8d009e0f,
            0xceb19bf2255445cd,
            0x0e746992d6d43a7c,
            0x41ed623b9dcc5fde,
            0x187a5a30d7c72edc,
            0x949ae2a9c1eb925a,
            0x7e9c76a7b7c35e68,
            0x4f96bf15b8309ff6,
            0x26c0c1fde233732e,
            0xb0453f72aa151615,
            0xf24b621a9ce9fece,
            0x99ed798408687b5f,
            0x3b13ec1221423b66,
            0xc67cf148a28afe59,
            0x22f7e0173f92e3fa,
            0x14186c5fda6683a0,
            0x97d608caa2603b2c,
            0xfde3b0bbba24ffa9,
            0xb7068eb48c472c77,
            0x9e34d72866b9fda0,
            0xbbb99c884cdef88e,
            0x81d3e01f472a8a1a,
            0xf84f506b3b60366d,
            0xfe3f42f01300db37,
            0xe385712a51c1f836,
            0x41dfd5e394245c79,
            0x60855dbedadb900a,
            0xbdb4c0aa38567476,
            0x9748802e8eec02cc,
            0x5ced256d257f88de,
            0x55acccdf9a80f155,
            0xa64b55b071afbbea,
            0xa205bfe6c724ce4d,
            0x69dd26ca8ac21744,
            0xef80e2ff2f6a9bc0,
            0xde266c0baa202c20,
            0xfa3463080ac74c50,
            0x379d968a40125c2b,
            0x4cbbd0a7b3c7d648,
            0xc92afd93f4c665d2,
            0x6e28f5adb7ae38dc,
            0x7c689c9c237be35e,
            0xaea41b29bd9d0f73,
            0x832cef631d77e59f,
            0x70cac8e87bc37dd3,
            0x8e8c98bbde68e764,
            0xd6117aeb3ddedded,
            0xd796ab808e766240,
            0x8953d0ea1a7d9814,
            0xa212eba4281b391c,
            0x21a555a8939ce597,
            0x809d31660f6d81a8,
            0x2356524b20ab400f,
            0x5bc611e1e49d0478,
            0xba9c065e2f385ce2,
            0xb0a0fd12f4e83899,
            0x14d076a35b1ff2ca,
            0x8acd0bb8cf9a93c0,
            0xe62e8ec094039ee4,
            0x38a536a7072bdc61,
            0xca256297602524f8,
            0xfc62ebfb3530caeb,
            0x8d8b0c05520569f6,
            0xbbaca65cf154c59d,
            0x3739b5ada7e338d3,
            0xdb9ea31f47365340,
            0x410b5c9c1da56755,
            0x7e0abc03dbd10283,
            0x136f87be70ed442e,
            0x6b727d4feddbe1e9,
            0x074ebb21183b01df,
            0x3fe92185b1985484,
            0xc5d8efd3c68305ca,
            0xd9bada21b17e272e,
            0x64d73133e1360f83,
            0xeb8563aa993e21f9,
            0xe5e8da50cceab28f,
            0x7a6f92eb3223d2f3,
            0xbdaf98370ea9b31b,
            0x1682a84457f077bc,
            0x4abd2d33b6e3be37,
            0xb35bc81a7c9d4c04,
            0x3e5bde3fb7cfe63d,
            0xff3abe6e2ffec974,
            0xb8116dd26cf6feec,
            0x7a77a6e4ed0cf081,
            0xb71eec2d5a184316,
            0x6fa932f77b4da817,
            0x795f79b33909b2c4,
            0x1b8755ef6b5eb34e,
            0x2255b72d7d6b2d79,
            0xf2bdafafa90bd50a,
            0x442a578f02cb1fc8,
            0xc25aefe55ecf83db,
            0x3114c056f9c5a676,
        ]
    }

    #[test]
    fn hash_golden_test() {
        if cfg!(target_endian = "big") || cfg!(target_pointer_width = "32") {
            eprintln!(
                "Skipping: golden data only maintained for little-endian 64-bit systems with int128"
            );
            return;
        }

        let k_golden = unsafe { get_k_golden() };
        for i in (0..K_NUM_GOLDEN_OUTPUTS).step_by(1) {
            print!("i = {}; input = {}", i, CASES[i].0);

            match general_purpose::STANDARD.decode(CASES[i].0) {
                Ok(binary) => {
                    assert!(binary.len() > 32);
                    let h =
                        combine_large_contiguous_on64bit_length_gt32(CASES[i].1, binary.as_slice());
                    assert_eq!(h, k_golden[i]);
                }
                Err(err) => panic!("{}", err),
            }
        }
    }
}
