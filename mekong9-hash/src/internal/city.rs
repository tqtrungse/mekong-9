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

use crate::internal::unaligned_access::{unaligned_load32, unaligned_load64};

/// Some primes between 2^63 and 2^64 for various uses.
const K0: u64 = 0xc3a5c85c97cb3127;
const K1: u64 = 0xb492b66fbe98f273;
const K2: u64 = 0x9ae16a3b2f90404f;

// Magic numbers for 32-bit hashing.  Copied from Murmur3.
const C1: u32 = 0xcc9e2d51;
const C2: u32 = 0x1b873593;

const MUL: u64 = 0x9ddfea08eb382d69;

macro_rules! permute3 {
    ($a: expr, $b: expr, $c: expr) => {
        core::mem::swap(&mut $a, &mut $b);
        core::mem::swap(&mut $a, &mut $c);
    };
}

#[cfg(target_endian = "little")]
#[inline]
fn u32_in_expected_order(v: u32) -> u32 {
    v
}

#[cfg(target_endian = "little")]
#[inline]
fn u64_in_expected_order(v: u64) -> u64 {
    v
}

#[cfg(target_endian = "big")]
#[inline]
fn u32_in_expected_order(v: u32) -> u32 {
    v.swap_bytes()
}

#[cfg(target_endian = "big")]
#[inline]
fn u64_in_expected_order(v: u64) -> u64 {
    v.swap_bytes()
}

#[inline]
fn fetch64(p: &[u8]) -> u64 {
    u64_in_expected_order(unaligned_load64(p.as_ptr()))
}

#[inline]
fn fetch32(p: &[u8]) -> u32 {
    u32_in_expected_order(unaligned_load32(p.as_ptr()))
}

/// Bitwise right rotate.  Normally this will compile to a single
/// instruction, especially if the shift is a manifest constant.
#[inline]
fn rotate64(val: u64, shift: i32) -> u64 {
    if shift == 0 {
        val
    } else {
        val.rotate_right(shift as u32)
    }
}

#[inline]
fn rotate32(val: u32, shift: i32) -> u32 {
    if shift == 0 {
        val
    } else {
        val.rotate_right(shift as u32)
    }
}

#[inline]
fn mur(mut a: u32, mut h: u32) -> u32 {
    // Helper from Murmur3 for combining two 32-bit values.
    a = a.wrapping_mul(C1);
    a = rotate32(a, 17);
    a = a.wrapping_mul(C2);
    h ^= a;
    h = rotate32(h, 19);
    h.wrapping_mul(5).wrapping_add(0xe6546b64)
}

#[inline]
fn fmix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

#[inline]
fn shift_mix(val: u64) -> u64 {
    val ^ (val >> 47)
}

#[inline]
fn hash64_len16(u: u64, v: u64, mul: u64) -> u64 {
    // Murmur-inspired hashing.
    let mut a = (u ^ v).wrapping_mul(mul);
    a ^= a >> 47;
    let mut b = (v ^ a).wrapping_mul(mul);
    b ^= b >> 47;
    b.wrapping_mul(mul)
}

fn hash32_len0_to4(data: &[u8]) -> u32 {
    let mut b = 0_u32;
    let mut c = 9_u32;
    let len = data.len();
    for i in (0..len).step_by(1) {
        let v = data[i] as i8;
        b = b.wrapping_mul(C1).wrapping_add(v as u32);
        c ^= b;
    }
    fmix(mur(b, mur(len as u32, c)))
}

#[inline]
fn hash64_len0_to16(data: &[u8]) -> u64 {
    let len = data.len();
    if len >= 8 {
        let mul = K2.wrapping_add(len.wrapping_mul(2) as u64);
        let a = fetch64(data).wrapping_add(K2);
        let b = fetch64(&data[len - 8..]);
        let c = rotate64(b, 37).wrapping_mul(mul).wrapping_add(a);
        let d = rotate64(a, 25).wrapping_add(b).wrapping_mul(mul);
        return hash64_len16(c, d, mul);
    }
    if len >= 4 {
        let mul = K2.wrapping_add(len.wrapping_mul(2) as u64);
        let a = fetch32(data) as u64;
        let v = hash64_len16(
            len.wrapping_add((a << 3) as usize) as u64,
            fetch32(&data[len - 4..]) as u64,
            mul,
        );
        return v;
    }
    if len > 0 {
        let a = data[0];
        let b = data[len >> 1];
        let c = data[len - 1];
        let y = (a as u32).wrapping_add((b as u32) << 8);
        let z = (len as u32).wrapping_add((c as u32) << 2);
        let k = (y as u64).wrapping_mul(K2) ^ (z as u64).wrapping_mul(K0);
        return shift_mix(k).wrapping_mul(K2);
    }
    K2
}

#[inline]
fn hash32_len5_to12(data: &[u8]) -> u32 {
    let len = data.len();
    let mut a = len as u32;
    let mut b = a.wrapping_mul(5);
    let mut c = 9_u32;
    let d = b;
    a = a.wrapping_add(fetch32(data));
    b = b.wrapping_add(fetch32(&data[len - 4..]));
    c = c.wrapping_add(fetch32(&data[((len >> 1) & 4)..]));
    fmix(mur(c, mur(b, mur(a, d))))
}

/// This probably works well for 16-byte strings as well, but it may be overkill
/// in that case.
#[inline]
fn hash64_len17_to32(data: &[u8]) -> u64 {
    let len = data.len();
    let mul = K2.wrapping_add(len.wrapping_mul(2) as u64);
    let a = fetch64(data).wrapping_mul(K1);
    let b = fetch64(&data[8..]);
    let c = fetch64(&data[len - 8..]).wrapping_mul(mul);
    let d = fetch64(&data[len - 16..]).wrapping_mul(K2);
    hash64_len16(
        rotate64(a.wrapping_add(b), 43)
            .wrapping_add(rotate64(c, 30))
            .wrapping_add(d),
        a.wrapping_add(rotate64(b.wrapping_add(K2), 18))
            .wrapping_add(c),
        mul,
    )
}

#[inline]
fn hash32_len13_to24(data: &[u8]) -> u32 {
    let len = data.len();
    let a = fetch32(&data[(len >> 1) - 4..]);
    let b = fetch32(&data[4..]);
    let c = fetch32(&data[len - 8..]);
    let d = fetch32(&data[len >> 1..]);
    let e = fetch32(data);
    let f = fetch32(&data[len - 4..]);
    let h = len as u32;
    fmix(mur(f, mur(e, mur(d, mur(c, mur(b, mur(a, h)))))))
}

/// Return an 8-byte hash for 33 to 64 bytes.
#[inline]
fn hash64_len33_to64(data: &[u8]) -> u64 {
    let len = data.len();
    let mul = K2.wrapping_add(len.wrapping_mul(2) as u64);
    let mut a = fetch64(data).wrapping_mul(K2);
    let b = fetch64(&data[8..]);
    let c = fetch64(&data[len - 24..]);
    let d = fetch64(&data[len - 32..]);
    let e = fetch64(&data[16..]).wrapping_mul(K2);
    let f = fetch64(&data[24..]).wrapping_mul(9);
    let g = fetch64(&data[len - 8..]);
    let h = fetch64(&data[len - 16..]).wrapping_mul(mul);

    let mut q = rotate64(b, 30).wrapping_add(c).wrapping_mul(9);

    let u = rotate64(a.wrapping_add(g), 43).wrapping_add(q);

    let v = (a.wrapping_add(g) ^ d).wrapping_add(f).wrapping_add(1);

    let w = u
        .wrapping_add(v)
        .wrapping_mul(mul)
        .swap_bytes()
        .wrapping_add(h);

    let x = rotate64(e.wrapping_add(f), 42).wrapping_add(c);

    let y = v
        .wrapping_add(w)
        .wrapping_mul(mul)
        .swap_bytes()
        .wrapping_add(g)
        .wrapping_mul(mul);

    let z = e.wrapping_add(f).wrapping_add(c);

    a = x
        .wrapping_add(z)
        .wrapping_mul(mul)
        .wrapping_add(y)
        .swap_bytes()
        .wrapping_add(b);

    q = z
        .wrapping_add(a)
        .wrapping_mul(mul)
        .wrapping_add(d)
        .wrapping_add(h);

    shift_mix(q).wrapping_mul(mul).wrapping_add(x)
}

/// Return a 16-byte hash for s[0] ... s[31], a, and b. Quick and dirty.
#[inline]
fn weak_hash64_len32_with_seeds(data: &[u8], a: u64, b: u64) -> (u64, u64) {
    let w = fetch64(data);
    let x = fetch64(&data[8..]);
    let y = fetch64(&data[16..]);
    let z = fetch64(&data[24..]);
    let mut aa = a.wrapping_add(w);
    let mut bb = rotate64(b.wrapping_add(aa).wrapping_add(z), 21);
    let c = aa;
    aa = aa.wrapping_add(x);
    aa = aa.wrapping_add(y);
    bb = bb.wrapping_add(rotate64(aa, 44));

    (aa.wrapping_add(z), bb.wrapping_add(c))
}

pub fn hash32(data: &[u8]) -> u32 {
    let len = data.len();
    if len <= 24 {
        return if len <= 12 {
            return if len <= 4 {
                hash32_len0_to4(data)
            } else {
                hash32_len5_to12(data)
            };
        } else {
            hash32_len13_to24(data)
        };
    }

    // len > 24
    let mut h = len as u32;
    let mut g = C1.wrapping_mul(h);
    let mut f = g;

    let a0 = rotate32(fetch32(&data[len - 4..]).wrapping_mul(C1), 17).wrapping_mul(C2);
    let a1 = rotate32(fetch32(&data[len - 8..]).wrapping_mul(C1), 17).wrapping_mul(C2);
    let a2 = rotate32(fetch32(&data[len - 16..]).wrapping_mul(C1), 17).wrapping_mul(C2);
    let a3 = rotate32(fetch32(&data[len - 12..]).wrapping_mul(C1), 17).wrapping_mul(C2);
    let a4 = rotate32(fetch32(&data[len - 20..]).wrapping_mul(C1), 17).wrapping_mul(C2);

    h ^= a0;
    h = rotate32(h, 19);
    h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
    h ^= a2;
    h = rotate32(h, 19);
    h = h.wrapping_mul(5).wrapping_add(0xe6546b64);

    g ^= a1;
    g = rotate32(g, 19);
    g = g.wrapping_mul(5).wrapping_add(0xe6546b64);
    g ^= a3;
    g = rotate32(g, 19);
    g = g.wrapping_mul(5).wrapping_add(0xe6546b64);

    f = f.wrapping_add(a4);
    f = rotate32(f, 19);
    f = f.wrapping_mul(5).wrapping_add(0xe6546b64);

    let mut iters = (len - 1) / 20;
    let mut seek = 0_usize;
    loop {
        let b0 = rotate32(fetch32(&data[seek..]).wrapping_mul(C1), 17).wrapping_mul(C2);
        let b1 = fetch32(&data[seek + 4..]);
        let b2 = rotate32(fetch32(&data[seek + 8..]).wrapping_mul(C1), 17).wrapping_mul(C2);
        let b3 = rotate32(fetch32(&data[seek + 12..]).wrapping_mul(C1), 17).wrapping_mul(C2);
        let b4 = fetch32(&data[seek + 16..]);

        h ^= b0;
        h = rotate32(h, 18);
        h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
        f = f.wrapping_add(b1);
        f = rotate32(f, 19);
        f = f.wrapping_mul(C1);
        g = g.wrapping_add(b2);
        g = rotate32(g, 18);
        g = g.wrapping_mul(5).wrapping_add(0xe6546b64);
        h ^= b3.wrapping_add(b1);
        h = rotate32(h, 19);
        h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
        g ^= b4;
        g = g.swap_bytes().wrapping_mul(5);
        h = h.wrapping_add(b4.wrapping_mul(5));
        h = h.swap_bytes();
        f = f.wrapping_add(b0);
        permute3!(f, h, g);
        seek += 20;

        iters -= 1;
        if iters == 0 {
            break;
        }
    }
    g = rotate32(g, 11).wrapping_mul(C1);
    g = rotate32(g, 17).wrapping_mul(C1);
    f = rotate32(f, 11).wrapping_mul(C1);
    f = rotate32(f, 17).wrapping_mul(C1);
    h = rotate32(h.wrapping_add(g), 19);
    h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
    h = rotate32(h, 17).wrapping_mul(C1);
    h = rotate32(h.wrapping_add(f), 19);
    h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
    h = rotate32(h, 17).wrapping_mul(C1);
    h
}

pub fn hash64(data: &[u8]) -> u64 {
    let mut len = data.len();
    if len <= 32 {
        return if len <= 16 {
            hash64_len0_to16(data)
        } else {
            hash64_len17_to32(data)
        };
    } else if len <= 64 {
        return hash64_len33_to64(data);
    }

    // For strings over 64 bytes we hash the end first, and then as we
    // loop we keep 56 bytes of state: v, w, x, y, and z.
    let mut x = fetch64(&data[len - 40..]);
    let mut y = fetch64(&data[len - 16..]).wrapping_add(fetch64(&data[len - 56..]));
    let mut z = hash64_len16(
        fetch64(&data[len - 48..]).wrapping_add(len as u64),
        fetch64(&data[len - 24..]),
        MUL,
    );
    let mut v = weak_hash64_len32_with_seeds(&data[len - 64..], len as u64, z);
    let mut w = weak_hash64_len32_with_seeds(&data[len - 32..], y.wrapping_add(K1), x);
    let mut seek = 0_usize;

    x = x.wrapping_mul(K1).wrapping_add(fetch64(data));

    // Decrease len to the nearest multiple of 64, and operate on 64-byte chunks.
    len = (len - 1) & !63;
    loop {
        let mut t = x
            .wrapping_add(y)
            .wrapping_add(v.0)
            .wrapping_add(fetch64(&data[seek + 8..]));
        x = rotate64(t, 37).wrapping_mul(K1);
        x ^= w.1;

        t = y
            .wrapping_add(v.1)
            .wrapping_add(fetch64(&data[seek + 48..]));
        y = rotate64(t, 42).wrapping_mul(K1);
        t = v.0.wrapping_add(fetch64(&data[seek + 40..]));
        y = y.wrapping_add(t);

        z = rotate64(z.wrapping_add(w.0), 33).wrapping_mul(K1);

        v = weak_hash64_len32_with_seeds(&data[seek..], v.1.wrapping_mul(K1), x.wrapping_add(w.0));

        w = weak_hash64_len32_with_seeds(
            &data[seek + 32..],
            z.wrapping_add(w.1),
            y.wrapping_add(fetch64(&data[seek + 16..])),
        );

        core::mem::swap(&mut z, &mut x);
        seek += 64;
        len -= 64;

        if len == 0 {
            break;
        }
    }

    hash64_len16(
        hash64_len16(v.0, w.0, MUL).wrapping_add(shift_mix(y).wrapping_mul(K1).wrapping_add(z)),
        hash64_len16(v.1, w.1, MUL).wrapping_add(x),
        MUL,
    )
}

#[inline]
pub fn hash64_with_seed(data: &[u8], seed: u64) -> u64 {
    hash64_with_seeds(data, K2, seed)
}

#[inline]
pub fn hash64_with_seeds(data: &[u8], seed0: u64, seed1: u64) -> u64 {
    hash64_len16(hash64(data).wrapping_sub(seed0), seed1, MUL)
}

#[cfg(test)]
mod test {
    use crate::internal::city::{hash32, hash64, hash64_with_seed, hash64_with_seeds};

    const K0: u64 = 0xc3a5c85c97cb3127;
    const K_SEED0: u64 = 1234567;
    const K_SEED1: u64 = K0;
    const K_DATA_SIZE: usize = 1 << 20;
    const K_TEST_SIZE: usize = 300;

    static TEST_DATA: &[[u64; 4]] = &[
        [0x9ae16a3b2f90404f, 0x75106db890237a4a, 0x3feac5f636039766, 0xdc56d17a],
        [0x541150e87f415e96, 0x1aef0d24b3148a1a, 0xbacc300e1e82345a, 0x99929334],
        [0x0f3786a4b25827c1, 0x34ee1a2bf767bd1c, 0x2f15ca2ebfb631f2, 0x4252edb7],
        [0xef923a7a1af78eab, 0x79163b1e1e9a9b18, 0xdf3b2aca6e1e4a30, 0xebc34f3c],
        [0x11df592596f41d88, 0x843ec0bce9042f9c, 0xcce2ea1e08b1eb30, 0x26f2b463],
        [0x831f448bdc5600b3, 0x62a24be3120a6919, 0x1b44098a41e010da, 0xb042c047],
        [0x3eca803e70304894, 0x0d80de767e4a920a, 0xa51cfbb292efd53d, 0xe73bb0a8],
        [0x1b5a063fb4c7f9f1, 0x318dbc24af66dee9, 0x10ef7b32d5c719af, 0x91dfdd75],
        [0xa0f10149a0e538d6, 0x69d008c20f87419f, 0x41b36376185b3e9e, 0xc87f95de],
        [0xfb8d9c70660b910b, 0xa45b0cc3476bff1b, 0xb28d1996144f0207, 0x3f5538ef],
        [0x236827beae282a46, 0xe43970221139c946, 0x4f3ac6faa837a3aa, 0x70eb1a1f],
        [0xc385e435136ecf7c, 0xd9d17368ff6c4a08, 0x1b31eed4e5251a67, 0xcfd63b83],
        [0xe3f6828b6017086d, 0x21b4d1900554b3b0, 0xbef38be1809e24f1, 0x894a52ef],
        [0x851fff285561dca0, 0x4d1277d73cdf416f, 0x28ccffa61010ebe2, 0x9cde6a54],
        [0x61152a63595a96d9, 0xd1a3a91ef3a7ba45, 0x443b6bb4a493ad0c, 0x6c4898d5],
        [0x44473e03be306c88, 0x30097761f872472a, 0x9fd1b669bfad82d7, 0x13e1978e],
        [0x03ead5f21d344056, 0xfb6420393cfb05c3, 0x407932394cbbd303, 0x051b4ba8],
        [0x6abbfde37ee03b5b, 0x83febf188d2cc113, 0xcda7b62d94d5b8ee, 0xb6b06e40],
        [0x0943e7ed63b3c080, 0x1ef207e9444ef7f8, 0xef4a9f9f8c6f9b4a, 0x0240a2f2],
        [0xd72ce05171ef8a1a, 0xc6bd6bd869203894, 0xc760e6396455d23a, 0x5dcefc30],
        [0x4182832b52d63735, 0x337097e123eea414, 0xb5a72ca0456df910, 0x7a48b105],
        [0xd6cdae892584a2cb, 0x58de0fa4eca17dcd, 0x43df30b8f5f1cb00, 0xfd55007b],
        [0x5c8e90bc267c5ee4, 0xe9ae044075d992d9, 0xf234cbfd1f0a1e59, 0x6b95894c],
        [0xbbd7f30ac310a6f3, 0xb23b570d2666685f, 0xfb13fb08c9814fe7, 0x3360e827],
        [0x36a097aa49519d97, 0x08204380a73c4065, 0x77c2004bdd9e276a, 0x45177e0b],
        [0x0dc78cb032c49217, 0x112464083f83e03a, 0x96ae53e28170c0f5, 0x7c6fffe4],
        [0x441593e0da922dfe, 0x936ef46061469b32, 0x204a1921197ddd87, 0xbbc78da4],
        [0x2ba3883d71cc2133, 0x72f2bbb32bed1a3c, 0x27e1bd96d4843251, 0xc5c25d39],
        [0xf2b6d2adf8423600, 0x7514e2f016a48722, 0x43045743a50396ba, 0xb6e5d06e],
        [0x38fffe7f3680d63c, 0xd513325255a7a6d1, 0x31ed47790f6ca62f, 0x6178504e],
        [0xb7477bf0b9ce37c6, 0x63b1c580a7fd02a4, 0x0f6433b9f10a5dac, 0xbd4c3637],
        [0x55bdb0e71e3edebd, 0xc7ab562bcf0568bc, 0x43166332f9ee684f, 0x6e7ac474],
        [0x0782fa1b08b475e7, 0xfb7138951c61b23b, 0x9829105e234fb11e, 0x1fb4b518],
        [0xc5dc19b876d37a80, 0x15ffcff666cfd710, 0xe8c30c72003103e2, 0x31d13d6d],
        [0x5e1141711d2d6706, 0xb537f6dee8de6933, 0x3af0a1fbbe027c54, 0x26fa72e3],
        [0x782edf6da001234f, 0x0f48cbd5c66c48f3, 0x808754d1e64e2a32, 0x6a7433bf],
        [0xd26285842ff04d44, 0x8f38d71341eacca9, 0x5ca436f4db7a883c, 0x4e6df758],
        [0xc6ab830865a6bae6, 0x6aa8e8dd4b98815c, 0xefe3846713c371e5, 0xd57f63ea],
        [0x044b3a1929232892, 0x061dca0e914fc217, 0xa607cc142096b964, 0x52ef73b3],
        [0x4b603d7932a8de4f, 0xfae64c464b8a8f45, 0x8fafab75661d602a, 0x03cb36c3],
        [0x4ec0b54cf1566aff, 0x30d2c7269b206bf4, 0x77c22e82295e1061, 0x72c39bea],
        [0xed8b7a4b34954ff7, 0x56432de31f4ee757, 0x85bd3abaa572b155, 0xa65aa25c],
        [0x5d28b43694176c26, 0x714cc8bc12d060ae, 0x3437726273a83fe6, 0x74740539],
        [0x6a1ef3639e1d202e, 0x919bc1bd145ad928, 0x30f3f7e48c28a773, 0xc3ae3c26],
        [0x159f4d9e0307b111, 0x03e17914a5675a0c, 0xaf849bd425047b51, 0xf29db8a2],
        [0xcc0a840725a7e25b, 0x57c69454396e193a, 0x976eaf7eee0b4540, 0x1ef4cbf4],
        [0xa2b27ee22f63c3f1, 0x9ebde0ce1b3976b2, 0x2fe6a92a257af308, 0xa9be6c41],
        [0xd8f2f234899bcab3, 0xb10b037297c3a168, 0xdebea2c510ceda7f, 0x0fa31801],
        [0x584f28543864844f, 0xd7cee9fc2d46f20d, 0xa38dca5657387205, 0x8331c5d8],
        [0xa94be46dd9aa41af, 0xa57e5b7723d3f9bd, 0x0034bf845a52fd2f, 0xe9876db8],
        [0x9a87bea227491d20, 0xa468657e2b9c43e7, 0xaf9ba60db8d89ef7, 0x27b0604e],
        [0x27688c24958d1a5c, 0xe3b4a1c9429cf253, 0x48a95811f70d64bc, 0xdcec07f2],
        [0x5d1d37790a1873ad, 0xed9cd4bcc5fa1090, 0xce51cde05d8cd96a, 0xcff0a82a],
        [0x1f03fd18b711eea9, 0x566d89b1946d381a, 0x6e96e83fc92563ab, 0xfec83621],
        [0xf0316f286cf527b6, 0xf84c29538de1aa5a, 0x7612ed3c923d4a71, 0x0743d8dc],
        [0x297008bcb3e3401d, 0x61a8e407f82b0c69, 0xa4a35bff0524fa0e, 0x64d41d26],
        [0x043c6252411ee3be, 0xb4ca1b8077777168, 0x2746dc3f7da1737f, 0xacd90c81],
        [0xce38a9a54fad6599, 0x6d6f4a90b9e8755e, 0xc3ecc79ff105de3f, 0x7c746a4b],
        [0x0270a9305fef70cf, 0x600193999d884f3a, 0x0f4d49eae09ed8a1, 0xb1047e99],
        [0xe71be7c28e84d119, 0xeb6ace59932736e6, 0x70c4397807ba12c5, 0xd1fd1068],
        [0xb5b58c24b53aaa19, 0xd2a6ab0773dd897f, 0xef762fe01ecb5b97, 0x56486077],
        [0x44dd59bd301995cf, 0x3ccabd76493ada1a, 0x540db4c87d55ef23, 0x6069be80],
        [0xb4d4789eb6f2630b, 0xbf6973263ce8ef0e, 0x0d1c75c50844b9d3, 0x2078359b],
        [0x12807833c463737c, 0x58e927ea3b3776b4, 0x72dd20ef1c2f8ad0, 0x9ea21004],
        [0xe88419922b87176f, 0xbcf32f41a7ddbf6f, 0xd6ebefd8085c1a0f, 0x9c9cfe88],
        [0x105191e0ec8f7f60, 0x5918dbfcca971e79, 0x6b285c8a944767b9, 0xb70a6ddd],
        [0xa5b88bf7399a9f07, 0xfca3ddfd96461cc4, 0xebe738fdc0282fc6, 0xdea37298],
        [0xd08c3f5747d84f50, 0x4e708b27d1b6f8ac, 0x70f70fd734888606, 0x8f480819],
        [0x2f72d12a40044b4b, 0x889689352fec53de, 0x0f03e6ad87eb2f36, 0x030b3b16],
        [0xaa1f61fdc5c2e11e, 0xc2c56cd11277ab27, 0xa1e73069fdf1f94f, 0xf31bc4e8],
        [0x9489b36fe2246244, 0x3355367033be74b8, 0x5f57c2277cbce516, 0x419f953b],
        [0x358d7c0476a044cd, 0xe0b7b47bcbd8854f, 0xffb42ec696705519, 0x20e9e76d],
        [0xb0c48df14275265a, 0x9da4448975905efa, 0xd716618e414ceb6d, 0x646f0ff8],
        [0xdaa70bb300956588, 0x410ea6883a240c6d, 0xf5c8239fb5673eb3, 0xeeb7eca8],
        [0x4ec97a20b6c4c7c2, 0x5913b1cd454f29fd, 0xa9629f9daf06d685, 0x08112bb9],
        [0x5c3323628435a2e8, 0x1bea45ce9e72a6e3, 0x904f0a7027ddb52e, 0x85a6d477],
        [0xc1ef26bea260abdb, 0x6ee423f2137f9280, 0xdf2118b946ed0b43, 0x56f76c84],
        [0x6be7381b115d653a, 0xed046190758ea511, 0xde6a45ffc3ed1159, 0x9af45d55],
        [0xae3eece1711b2105, 0x14fd3f4027f81a4a, 0xabb7e45177d151db, 0xd1c33760],
        [0x376c28588b8fb389, 0x6b045e84d8491ed2, 0x4e857effb7d4e7dc, 0xc56bbf69],
        [0x58d943503bb6748f, 0x419c6c8e88ac70f6, 0x586760cbf3d3d368, 0xabecfb9b],
        [0xdfff5989f5cfd9a1, 0xbcee2e7ea3a96f83, 0x681c7874adb29017, 0x8de13255],
        [0x7fb19eb1a496e8f5, 0xd49e5dfdb5c0833f, 0xc0d5d7b2f7c48dc7, 0xa98ee299],
        [0x5dba5b0dadccdbaa, 0x4ba8da8ded87fcdc, 0xf693fdd25badf2f0, 0x3015f556],
        [0x688bef4b135a6829, 0x8d31d82abcd54e8e, 0xf95f8a30d55036d7, 0x5a430e29],
        [0xd8323be05433a412, 0x8d48fa2b2b76141d, 0x3d346f23978336a5, 0x2797add0],
        [0x3b5404278a55a7fc, 0x23ca0b327c2d0a81, 0xa6d65329571c892c, 0x27d55016],
        [0x2a96a3f96c5e9bbc, 0x8caf8566e212dda8, 0x904de559ca16e45e, 0x84945a82],
        [0x22bebfdcc26d18ff, 0x4b4d8dcb10807ba1, 0x40265eee30c6b896, 0x3ef7e224],
        [0x627a2249ec6bbcc2, 0xc0578b462a46735a, 0x4974b8ee1c2d4f1f, 0x35ed8dc8],
        [0x3abaf1667ba2f3e0, 0x0ee78476b5eeadc1, 0x7e56ac0a6ca4f3f4, 0x6a75e43d],
        [0x3931ac68c5f1b2c9, 0xefe3892363ab0fb0, 0x40b707268337cd36, 0x235d9805],
        [0xb98fb0606f416754, 0x46a6e5547ba99c1e, 0x0c909d82112a8ed2, 0xf7d69572],
        [0x7f7729a33e58fcc4, 0x2e4bc1e7a023ead4, 0xe707008ea7ca6222, 0xbacd0199],
        [0x42a0aa9ce82848b3, 0x57232730e6bee175, 0xf89bb3f370782031, 0xe428f50e],
        [0x6b2c6d38408a4889, 0xde3ef6f68fb25885, 0x20754f456c203361, 0x81eaaad3],
        [0x930380a3741e862a, 0x348d28638dc71658, 0x89dedcfd1654ea0d, 0xaddbd3e3],
        [0x94808b5d2aa25f9a, 0xcec72968128195e0, 0xd9f4da2bdc1e130f, 0xe66dbca0],
        [0xb31abb08ae6e3d38, 0x9eb9a95cbd9e8223, 0x8019e79b7ee94ea9, 0xafe11fd5],
        [0xdccb5534a893ea1a, 0xce71c398708c6131, 0xfe2396315457c164, 0xa71a406f],
        [0x6369163565814de6, 0x8feb86fb38d08c2f, 0x4976933485cc9a20, 0x9d90eaf5],
        [0xedee4ff253d9f9b3, 0x96ef76fb279ef0ad, 0xa4d204d179db2460, 0x6665db10],
        [0x941993df6e633214, 0x929bc1beca5b72c6, 0x141fc52b8d55572d, 0x9c977cbf],
        [0x859838293f64cd4c, 0x484403b39d44ad79, 0xbf674e64d64b9339, 0xee83ddd4],
        [0xc19b5648e0d9f555, 0x328e47b2b7562993, 0xe756b92ba4bd6a51, 0x026519cc],
        [0xf963b63b9006c248, 0x9e9bf727ffaa00bc, 0xc73bacc75b917e3a, 0xa485a53f],
        [0x6a8aa0852a8c1f3b, 0xc8f1e5e206a21016, 0x2aa554aed1ebb524, 0xf62bc412],
        [0x740428b4d45e5fb8, 0x4c95a4ce922cb0a5, 0xe99c3ba78feae796, 0x8975a436],
        [0x658b883b3a872b86, 0x2f0e303f0f64827a, 0x0975337e23dc45e1, 0x94ff7f41],
        [0x6df0a977da5d27d4, 0x0891dd0e7cb19508, 0xfd65434a0b71e680, 0x760aa031],
        [0xa900275464ae07ef, 0x11f2cfda34beb4a3, 0x09abf91e5a1c38e4, 0x3bda76df],
        [0x810bc8aa0c40bcb0, 0x448a019568d01441, 0xf60ec52f60d3aeae, 0x498e2e65],
        [0x22036327deb59ed7, 0xadc05ceb97026a02, 0x48bff0654262672b, 0xd38deb48],
        [0x7d14dfa9772b00c8, 0x595735efc7eeaed7, 0x29872854f94c3507, 0x82b3fb6b],
        [0x2d777cddb912675d, 0x278d7b10722a13f9, 0xf5c02bfb7cc078af, 0xe500e25f],
        [0xf2ec98824e8aa613, 0x5eb7e3fb53fe3bed, 0x12c22860466e1dd4, 0xbd2bb07c],
        [0x5e763988e21f487f, 0x24189de8065d8dc5, 0xd1519d2403b62aa0, 0x3a2b431d],
        [0x48949dc327bb96ad, 0xe1fd21636c5c50b4, 0x3f6eb7f13a8712b4, 0x7322a83d],
        [0xb7c4209fb24a85c5, 0xb35feb319c79ce10, 0xf0d3de191833b922, 0xa645ca1c],
        [0x9c9e5be0943d4b05, 0xb73dc69e45201cbb, 0xaab17180bfe5083d, 0x8909a45a],
        [0x3898bca4dfd6638d, 0xf911ff35efef0167, 0x24bdf69e5091fc88, 0xbd30074c],
        [0x5b5d2557400e68e7, 0x098d610033574cee, 0xdfd08772ce385deb, 0xc17cf001],
        [0xa927ed8b2bf09bb6, 0x606e52f10ae94eca, 0x71c2203feb35a9ee, 0x26ffd25a],
        [0x8d25746414aedf28, 0x34b1629d28b33d3a, 0x4d5394aea5f82d7b, 0xf1d8ce3c],
        [0xb5bbdb73458712f2, 0x1ff887b3c2a35137, 0x7f7231f702d0ace9, 0x3ee8fb17],
        [0x3d32a26e3ab9d254, 0xfc4070574dc30d3a, 0xf02629579c2b27c9, 0xa77acc2a],
        [0x9371d3c35fa5e9a5, 0x0042967cf4d01f30, 0x652d1eeae704145c, 0xf4556dee],
        [0xcbaa3cb8f64f54e0, 0x76c3b48ee5c08417, 0x09f7d24e87e61ce9, 0xde287a64],
        [0xb2e23e8116c2ba9f, 0x7e4d9c0060101151, 0x3310da5e5028f367, 0x878e55b9],
        [0x8aa77f52d7868eb9, 0x4d55bd587584e6e2, 0x0d2db37041f495f5, 0x07648486],
        [0x858fea922c7fe0c3, 0xcfe8326bf733bc6f, 0x4e5e2018cf8f7dfc, 0x57ac0fb1],
        [0x46ef25fdec8392b1, 0xe48d7b6d42a5cd35, 0x56a6fe1c175299ca, 0xd01967ca],
        [0x8d078f726b2df464, 0xb50ee71cdcabb299, 0xf4af300106f9c7ba, 0x96ecdf74],
        [0x35ea86e6960ca950, 0x34fe1fe234fc5c76, 0xa00207a3dc2a72b7, 0x779f5506],
        [0x8aee9edbc15dd011, 0x51f5839dc8462695, 0xb2213e17c37dca2d, 0x3c94c2de],
        [0xc3e142ba98432dda, 0x911d060cab126188, 0xb753fbfa8365b844, 0x39f98faf],
        [0x123ba6b99c8cd8db, 0x448e582672ee07c4, 0xcebe379292db9e65, 0x7af31199],
        [0xba87acef79d14f53, 0xb3e0fcae63a11558, 0xd5ac313a593a9f45, 0xe341a9d6],
        [0x0bcd3957d5717dc3, 0x2da746741b03a007, 0x873816f4b1ece472, 0xca24aeeb],
        [0x61442ff55609168e, 0x6447c5fc76e8c9cf, 0x6a846de83ae15728, 0xb2252b57],
        [0xdbe4b1b2d174757f, 0x506512da18712656, 0x06857f3e0b8dd95f, 0x72c81da1],
        [0x531e8e77b363161c, 0xeece0b43e2dae030, 0x8294b82c78f34ed1, 0x6b9fce95],
        [0xf71e9c926d711e2b, 0xd77af2853a4ceaa1, 0x9aa0d6d76a36fae7, 0x19399857],
        [0xcb20ac28f52df368, 0xe6705ee7880996de, 0x9b665cc3ec6972f2, 0x3c57a994],
        [0xe4a794b4acb94b55, 0x89795358057b661b, 0x9c4cdcec176d7a70, 0xc053e729],
        [0xcb942e91443e7208, 0xe335de8125567c2a, 0xd4d74d268b86df1f, 0x51cbbba7],
        [0xecca7563c203f7ba, 0x177ae2423ef34bb2, 0xf60b7243400c5731, 0x1acde79a],
        [0x1652cb940177c8b5, 0x8c4fe7d85d2a6d6d, 0xf6216ad097e54e72, 0x2d160d13],
        [0x31fed0fc04c13ce8, 0x3d5d03dbf7ff240a, 0x727c5c9b51581203, 0x787f5801],
        [0xe7b668947590b9b3, 0xbaa41ad32938d3fa, 0xabcbc8d4ca4b39e4, 0xc9629828],
        [0x1de2119923e8ef3c, 0x6ab27c096cf2fe14, 0x8c3658edca958891, 0xbe139231],
        [0x1269df1e69e14fa7, 0x992f9d58ac5041b7, 0xe97fcf695a7cbbb4, 0x7df699ef],
        [0x820826d7aba567ff, 0x1f73d28e036a52f3, 0x41c4c5a73f3b0893, 0x8ce6b96d],
        [0xffe0547e4923cef9, 0x3534ed49b9da5b02, 0x548a273700fba03d, 0x6f9ed99c],
        [0x72da8d1b11d8bc8b, 0xba94b56b91b681c6, 0x4e8cc51bd9b0fc8c, 0xe0244796],
        [0xd62ab4e3f88fc797, 0xea86c7aeb6283ae4, 0x0b5b93e09a7fe465, 0x4ccf7e75],
        [0xd0f06c28c7b36823, 0x1008cb0874de4bb8, 0xd6c7ff816c7a737b, 0x915cef86],
        [0x99b7042460d72ec6, 0x2a53e5e2b8e795c2, 0x53a78132d9e1b3e3, 0x5cb59482],
        [0x4f4dfcfc0ec2bae5, 0x841233148268a1b8, 0x09248a76ab8be0d3, 0x6ca3f532],
        [0xfe86bf9d4422b9ae, 0xebce89c90641ef9c, 0x1c84e2292c0b5659, 0xe24f3859],
        [0xa90d81060932dbb0, 0x8acfaa88c5fbe92b, 0x7c6f3447e90f7f3f, 0xadf5a9c7],
        [0x17938a1b0e7f5952, 0x22cadd2f56f8a4be, 0x84b0d1183d5ed7c1, 0x32264b75],
        [0xde9e0cb0e16f6e6d, 0x238e6283aa4f6594, 0x4fb9c914c2f0a13b, 0xa64b3376],
        [0x6d4b876d9b146d1a, 0xaab2d64ce8f26739, 0xd315f93600e83fe5, 0x0d33890e],
        [0xe698fa3f54e6ea22, 0xbd28e20e7455358c, 0x9ace161f6ea76e66, 0x926d4b63],
        [0x7bc0deed4fb349f7, 0x1771aff25dc722fa, 0x19ff0644d9681917, 0xd51ba539],
        [0xdb4b15e88533f622, 0x256d6d2419b41ce9, 0x9d7c5378396765d5, 0x7f37636d],
        [0x922834735e86ecb2, 0x363382685b88328e, 0xe9c92960d7144630, 0xb98026c0],
        [0x30f1d72c812f1eb8, 0xb567cd4a69cd8989, 0x820b6c992a51f0bc, 0xb877767e],
        [0x168884267f3817e9, 0x5b376e050f637645, 0x1c18314abd34497a, 0x0aefae77],
        [0x82e78596ee3e56a7, 0x25697d9c87f30d98, 0x7600a8342834924d, 0x0f686911],
        [0xaa2d6cf22e3cc252, 0x9b4dec4f5e179f16, 0x76fb0fba1d99a99a, 0x3deadf12],
        [0x7bf5ffd7f69385c7, 0xfc077b1d8bc82879, 0x9c04e36f9ed83a24, 0xccf02a4e],
        [0xe89c8ff9f9c6e34b, 0xf54c0f669a49f6c4, 0xfc3e46f5d846adef, 0x176c1722],
        [0xa18fbcdccd11e1f4, 0x8248216751dfd65e, 0x40c089f208d89d7c, 0x026f82ad],
        [0x2d54f40cc4088b17, 0x59d15633b0cd1399, 0xa8cc04bb1bffd15b, 0xb5244f42],
        [0x69276946cb4e87c7, 0x62bdbe6183be6fa9, 0x3ba9773dac442a1a, 0x49a689e5],
        [0x668174a3f443df1d, 0x407299392da1ce86, 0xc2a3f7d7f2c5be28, 0x059fcdd3],
        [0x05e29be847bd5046, 0xb561c7f19c8f80c3, 0x5e5abd5021ccaeaf, 0x4f4b04e9],
        [0xcd0d79f2164da014, 0x4c386bb5c5d6ca0c, 0x8e771b03647c3b63, 0x8b00f891],
        [0xe0e6fc0b1628af1d, 0x29be5fb4c27a2949, 0x1c3f781a604d3630, 0x16e114f3],
        [0x2058927664adfd93, 0x6e8f968c7963baa5, 0xaf3dced6fff7c394, 0xd6b6dadc],
        [0xdc107285fd8e1af7, 0xa8641a0609321f3f, 0xdb06e89ffdc54466, 0x897e20ac],
        [0xfbba1afe2e3280f1, 0x0755a5f392f07fce, 0x9e44a9a15402809a, 0xf996e05d],
        [0xbfa10785ddc1011b, 0xb6e1c4d2f670f7de, 0x517d95604e4fcc1f, 0xc4306af6],
        [0x534cc35f0ee1eb4e, 0xb703820f1f3b3dce, 0x0884aa164cf22363, 0x6dcad433],
        [0x07ca6e3933995dac, 0x0fd118c77daa8188, 0x3aceb7b5e7da6545, 0x3c07374d],
        [0xf0d6044f6efd7598, 0xe044d6ba4369856e, 0x91968e4f8c8a1a4c, 0xf0f4602c],
        [0x3d69e52049879d61, 0x76610636ea9f74fe, 0xe9bf5602f89310c0, 0x3e1ea071],
        [0x79da242a16acae31, 0x0183c5f438e29d40, 0x6d351710ae92f3de, 0x67580f0c],
        [0x461c82656a74fb57, 0xd84b491b275aa0f7, 0x8f262cb29a6eb8b2, 0x4e109454],
        [0x053c1a66d0b13003, 0x731f060e6fe797fc, 0xdaa56811791371e3, 0x88a474a7],
        [0x0d3a2efec0f047e9, 0x1cabce58853e58ea, 0x7a17b2eae3256be4, 0x05b5bedd],
        [0x43c64d7484f7f9b2, 0x5da002b64aafaeb7, 0xb576c1e45800a716, 0x1aaddfa7],
        [0xa7dec6ad81cf7fa1, 0x180c1ab708683063, 0x95e0fd7008d67cff, 0x5be07fd8],
        [0x05408a1df99d4aff, 0xb9565e588740f6bd, 0xabf241813b08006e, 0xcbca8606],
        [0xa8b27a6bcaeeed4b, 0xaec1eeded6a87e39, 0x9daf246d6fed8326, 0xbde64d01],
        [0x9a952a8246fdc269, 0xd0dcfcac74ef278c, 0x250f7139836f0f1f, 0xee90cf33],
        [0xc930841d1d88684f, 0x5eb66eb18b7f9672, 0xe455d413008a2546, 0x4305c3ce],
        [0x94dc6971e3cf071a, 0x994c7003b73b2b34, 0x0ea16e85978694e5, 0x4b3a1d76],
        [0x07fc98006e25cac9, 0x77fee0484cda86a7, 0x376ec3d447060456, 0xa8bb6d80],
        [0x0bd781c4454103f6, 0x612197322f49c931, 0xb9cf17fd7e5462d5, 0x1f9fa607],
        [0xda60e6b14479f9df, 0x3bdccf69ece16792, 0x18ebf45c4fecfdc9, 0x8d0e4ed2],
        [0x04ca56a348b6c4d3, 0x60618537c3872514, 0x2fbb9f0e65871b09, 0x1bf31347],
        [0xebd22d4b70946401, 0x6863602bf7139017, 0xc0b1ac4e11b00666, 0x1ae3fc5b],
        [0x03cc4693d6cbcb0c, 0x0501689ea1c70ffa, 0x10a4353e9c89e364, 0x459c3930],
        [0x38908e43f7ba5ef0, 0x1ab035d4e7781e76, 0x41d133e8c0a68ff7, 0xe00c4184],
        [0x34983ccc6aa40205, 0x21802cad34e72bc4, 0x01943e8fb3c17bb8, 0xffc7a781],
        [0x86215c45dcac9905, 0xea546afe851cae4b, 0xd85b6457e489e374, 0x6a125480],
        [0x420fc255c38db175, 0xd503cd0f3c1208d1, 0xd4684e74c825a0bc, 0x88a1512b],
        [0x1d7a31f5bc8fe2f9, 0x4763991092dcf836, 0xed695f55b97416f4, 0x549bbbe5],
        [0x94129a84c376a26e, 0xc245e859dc231933, 0x1b8f74fecf917453, 0xc133d38c],
        [0x1d3a9809dab05c8d, 0x0adddeb4f71c93e8, 0x0ef342eb36631edb, 0xfcace348],
        [0x90fa3ccbd60848da, 0xdfa6e0595b569e11, 0xe585d067a1f5135d, 0xed7b6f9a],
        [0x2dbb4fc71b554514, 0x9650e04b86be0f82, 0x60f2304fba9274d3, 0x6d907dda],
        [0xb98bf4274d18374a, 0x1b669fd4c7f9a19a, 0xb1f5972b88ba2b7a, 0x7a4d48d5],
        [0xd6781d0b5e18eb68, 0xb992913cae09b533, 0x58f6021caaee3a40, 0xe686f3db],
        [0x226651cf18f4884c, 0x595052a874f0f51c, 0xc9b75162b23bab42, 0x0cce7c55],
        [0xa734fb047d3162d6, 0xe523170d240ba3a5, 0x125a6972809730e8, 0x0f58b96b],
        [0xc6df6364a24f75a3, 0xc294e2c84c4f5df8, 0xa88df65c6a89313b, 0x1bbf6f60],
        [0x0d8d1364c1fbcd10, 0x2d7cc7f54832deaa, 0x4e22c876a7c57625, 0xce5e0cc2],
        [0xaae06f9146db885f, 0x3598736441e280d9, 0xfba339b117083e55, 0x584cfd6f],
        [0x8955ef07631e3bcc, 0x7d70965ea3926f83, 0x39aed4134f8b2db6, 0x8f9bbc33],
        [0xad611c609cfbe412, 0xd3c00b18bf253877, 0x90b2172e1f3d0bfd, 0xd7640d95],
        [0xd5339adc295d5d69, 0xb633cc1dcb8b586a, 0xee84184cf5b1aeaf, 0x03d12a2b],
        [0x40d0aeff521375a8, 0x77ba1ad7ecebd506, 0x547c6f1a7d9df427, 0xaaeafed0],
        [0x8b2d54ae1a3df769, 0x11e7adaee3216679, 0x3483781efc563e03, 0x95b9b814],
        [0x99c175819b4eae28, 0x932e8ff9f7a40043, 0xec78dcab07ca9f7c, 0x45fbe66e],
        [0x2a418335779b82fc, 0xaf0295987849a76b, 0xc12bc5ff0213f46e, 0xb4baa7a8],
        [0x3b1fc6a3d279e67d, 0x070ea1e49c226396, 0x25505adcf104697c, 0x83e962fe],
        [0xd97eacdf10f1c3c9, 0xb54f4654043a36e0, 0x0b128f6eb09d1234, 0xaac3531c],
        [0x293a5c1c4e203cd4, 0x6b3329f1c130cefe, 0xf2e32f8ec76aac91, 0x2b1db7cc],
        [0x4290e018ffaedde7, 0xa14948545418eb5e, 0x72d851b202284636, 0xcf00cd31],
        [0xf919a59cbde8bf2f, 0xa56d04203b2dc5a5, 0x38b06753ac871e48, 0x7d3c43b8],
        [0x1d70a3f5521d7fa4, 0xfb97b3fdc5891965, 0x299d49bbbe3535af, 0xcbd5fac6],
        [0x6af98d7b656d0d7c, 0xd2e99ae96d6b5c0c, 0xf63bd1603ef80627, 0x76d0fec4],
        [0x395b7a8adb96ab75, 0x0582df7165b20f4a, 0xe52bd30e9ff657f9, 0x405e3402],
        [0x3822dd82c7df012f, 0xb9029b40bd9f122b, 0xfd25b988468266c4, 0xc732c481],
        [0x79f7efe4a80b951a, 0xdd3a3fddfc6c9c41, 0xab4c812f9e27aa40, 0xa8d123c9],
        [0xae6e59f5f055921a, 0x000e9d9b7bf68e82, 0x5ce4e4a5b269cc59, 0x1e80ad7d],
        [0x8959dbbf07387d36, 0xb4658afce48ea35d, 0x8f3f82437d8cb8d6, 0x52aeb863],
        [0x4739613234278a49, 0x99ea5bcd340bf663, 0x258640912e712b12, 0xef7c0c18],
        [0x420e6c926bc54841, 0x96dbbf6f4e7c75cd, 0xd8d40fa70c3c67bb, 0xb6ad4b68],
        [0xc8601bab561bc1b7, 0x72b26272a0ff869a, 0x56fdfc986d6bc3c4, 0xc1e46b17],
        [0xb2d294931a0e20eb, 0x284ffd9a0815bc38, 0x01f8a103aac9bbe6, 0x57b8df25],
        [0x7966f53c37b6c6d7, 0x8e6abcfb3aa2b88f, 0x7f2e5e0724e5f345, 0xe9fa36d6],
        [0xbe9bb0abd03b7368, 0x13bca93a3031be55, 0xe864f4f52b55b472, 0x8f8daefc],
        [0xa08d128c5f1649be, 0xa8166c3dbbe19aad, 0xcb9f914f829ec62c, 0x06e1bb7e],
        [0x7c386f0ffe0465ac, 0x530419c9d843dbf3, 0x7450e3a4f72b8d8c, 0xfd0076f0],
        [0x0bb362094e7ef4f8, 0xff3c2a48966f9725, 0x55152803acd4a7fe, 0x899b17b6],
        [0xcd80dea24321eea4, 0x52b4fdc8130c2b15, 0xf3ea100b154bfb82, 0xe3e84e31],
        [0xd599a04125372c3a, 0x313136c56a56f363, 0x1e993c3677625832, 0xeef79b6b],
        [0x0dbbf541e9dfda0a, 0x1479fceb6db4f844, 0x31ab576b59062534, 0x868e3315],
        [0xc2ee3288be4fe2bf, 0x0c65d2f5ddf32b92, 0xaf6ecdf121ba5485, 0x4639a426],
        [0xd86603ced1ed4730, 0xf9de718aaada7709, 0xdb8b9755194c6535, 0xf3213646],
        [0x915263c671b28809, 0xa815378e7ad762fd, 0xabec6dc9b669f559, 0x17f148e9],
        [0x2b67cdd38c307a5e, 0x0cb1d45bb5c9fe1c, 0x800baf2a02ec18ad, 0xbfd94880],
        [0x2d107419073b9cd0, 0xa96db0740cef8f54, 0xec41ee91b3ecdc1b, 0xbb1fa7f3],
        [0xf3e9487ec0e26dfc, 0x1ab1f63224e837fa, 0x119983bb5a8125d8, 0x088816b1],
        [0x1160987c8fe86f7d, 0x879e6db1481eb91b, 0xd7dcb802bfe6885d, 0x5c2faeb3],
        [0xeab8112c560b967b, 0x97f550b58e89dbae, 0x846ed506d304051f, 0x51b5fc6f],
        [0x1addcf0386d35351, 0xb5f436561f8f1484, 0x85d38e22181c9bb1, 0x33d94752],
        [0xd445ba84bf803e09, 0x1216c2497038f804, 0x2293216ea2237207, 0xb0c92948],
        [0x37235a096a8be435, 0xd9b73130493589c2, 0x3b1024f59378d3be, 0xc7171590],
        [0x763ad6ea2fe1c99d, 0xcf7af5368ac1e26b, 0x4d5e451b3bb8d3d4, 0x240a67fb],
        [0xea627fc84cd1b857, 0x85e372494520071f, 0x69ec61800845780b, 0xe1843cd5],
        [0x1f2ffd79f2cdc0c8, 0x726a1bc31b337aaa, 0x678b7f275ef96434, 0xfda1452b],
        [0x39a9e146ec4b3210, 0xf63f75802a78b1ac, 0xe2e22539c94741c3, 0xa2cad330],
        [0x74cba303e2dd9d6d, 0x692699b83289fad1, 0xdfb9aa7874678480, 0x53467e16],
        [0x4cbc2b73a43071e0, 0x56c5db4c4ca4e0b7, 0x1b275a162f46bd3d, 0xda14a8d0],
        [0x875638b9715d2221, 0xd9ba0615c0c58740, 0x616d4be2dfe825aa, 0x67333551],
        [0xfb686b2782994a8d, 0xedee60693756bb48, 0xe6bc3cae0ded2ef5, 0xa0ebd66e],
        [0xab21d81a911e6723, 0x4c31b07354852f59, 0x835da384c9384744, 0x4b769593],
        [0x33d013cc0cd46ecf, 0x3de726423aea122c, 0x116af51117fe21a9, 0x6aa75624],
        [0x8ca92c7cd39fae5d, 0x0317e620e1bf20f1, 0x4f0b33bf2194b97f, 0x602a3f96],
        [0xfdde3b03f018f43e, 0x038f932946c78660, 0xc84084ce946851ee, 0xcd183c4d],
        [0x9c8502050e9c9458, 0xd6d2a1a69964beb9, 0x1675766f480229b5, 0x960a4d07],
        [0x348176ca2fa2fdd2, 0x3a89c514cc360c2d, 0x9f90b8afb318d6d0, 0x9ae998c4],
        [0x4a3d3dfbbaea130b, 0x4e221c920f61ed01, 0x553fd6cd1304531f, 0x74e2179d],
        [0xb371f768cdf4edb9, 0xbdef2ace6d2de0f0, 0xe05b4100f7f1baec, 0xee9bae25],
        [0x07a1d2e96934f61f, 0xeb1760ae6af7d961, 0x887eb0da063005df, 0xb66edf10],
        [0x8be53d466d4728f2, 0x86a5ac8e0d416640, 0x984aa464cdb5c8bb, 0xd6209737],
        [0x829677eb03abf042, 0x043cad004b6bc2c0, 0xf2f224756803971a, 0x0b994a88],
        [0x0754435bae3496fc, 0x5707fc006f094dcf, 0x8951c86ab19d8e40, 0xa05d43c0],
        [0xfda9877ea8e3805f, 0x31e868b6ffd521b7, 0xb08c90681fb6a0fd, 0xc79f73a8],
        [0x2e36f523ca8f5eb5, 0x8b22932f89b27513, 0x331cd6ecbfadc1bb, 0xa490aff5],
        [0x21a378ef76828208, 0xa5c13037fa841da2, 0x506d22a53fbe9812, 0xdfad65b4],
        [0xccdd5600054b16ca, 0xf78846e84204cb7b, 0x1f9faec82c24eac9, 0x01d07dfb],
        [0x7854468f4e0cabd0, 0x3a3f6b4f098d0692, 0xae2423ec7799d30d, 0x416df9a0],
        [0x7f88db5346d8f997, 0x88eac9aacc653798, 0x68a4d0295f8eefa1, 0x1f8fb9cc],
        [0xbb3fb5fb01d60fcf, 0x1b7cc0847a215eb6, 0x1246c994437990a1, 0x7abf48e3],
        [0x2e783e1761acd84d, 0x39158042bac975a0, 0x1cd21c5a8071188d, 0xdea4e3dd],
        [0x392058251cf22acc, 0x944ec4475ead4620, 0xb330a10b5cb94166, 0xc6064f22],
        [0xadf5c1e5d6419947, 0x2a9747bc659d28aa, 0x095c5b8cb1f5d62c, 0x743bed9c],
        [0x6bc1db2c2bee5aba, 0xe63b0ed635307398, 0x7b2eca111f30dbbc, 0xfce254d5],
        [0xb00f898229efa508, 0x83b7590ad7f6985c, 0x2780e70a0592e41d, 0xe47ec9d1],
        [0xb56eb769ce0d9a8c, 0xce196117bfbcaf04, 0xb26c3c3797d66165, 0x334a145c],
        [0x70c0637675b94150, 0x259e1669305b0a15, 0x46e1dd9fd387a58d, 0xadec1e3c],
        [0x74c0b8a6821faafe, 0xabac39d7491370e7, 0xfaf0b2a48a4e6aed, 0xf6a9fbf8],
        [0x5fb5e48ac7b7fa4f, 0xa96170f08f5acbc7, 0xbbf5c63d4f52a1e5, 0x5398210c],
    ];

    fn test_unchanging(expected: &[u64; 4], data: &[u8], offset: usize, len: usize) {
        let l = offset + len;
        assert_eq!(expected[0], hash64(&data[offset..l]));
        assert_eq!(expected[3], hash32(&data[offset..l]) as u64);
        assert_eq!(expected[1], hash64_with_seed(&data[offset..l], K_SEED0));
        assert_eq!(
            expected[2],
            hash64_with_seeds(&data[offset..l], K_SEED0, K_SEED1)
        );
    }

    #[test]
    fn test_city_hash() {
        let mut a = 9_u64;
        let mut b = 777_u64;
        let mut i = 0_usize;
        let mut data: [u8; K_DATA_SIZE] = [0; K_DATA_SIZE];
        while i < K_DATA_SIZE {
            a = a.wrapping_add(b);
            b = b.wrapping_add(a);
            a = (a ^ (a >> 41)).wrapping_mul(K0);
            b = (b ^ (b >> 41)).wrapping_mul(K0).wrapping_add(i as u64);
            data[i] = (b >> 37) as u8;
            i += 1;
        }

        i = 0;
        while i < K_TEST_SIZE - 1 {
            test_unchanging(&TEST_DATA[i], &data, i * i, i);
            i += 1;
        }
        test_unchanging(&TEST_DATA[i], &data, 0, K_DATA_SIZE);
    }
}
