//! IEEE binary16 conversions (round to nearest even), as the GPUs do.

pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut mant = x & 0x7f_ffff;
    let e = ((x >> 23) & 0xff) as i32;
    if e == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let exp = e - 127 + 15;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        mant |= 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut h = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mut mant = (h & 0x3ff) as u32;
    let x = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e: i32 = 1;
            while mant & 0x400 == 0 {
                mant <<= 1;
                e -= 1;
            }
            sign | (((e + 127 - 15) as u32) << 23) | ((mant & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(x)
}

/// `x` rounded to the nearest f16 value.
pub fn round_f16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_rounds() {
        for x in [0.0f32, 1.0, -2.5, 65504.0, 6.1e-5, 1e-7, 3.140625] {
            let h = f32_to_f16(x);
            assert_eq!(f32_to_f16(f16_to_f32(h)), h);
        }
        assert_eq!(round_f16(1.0 + 1.0 / 4096.0), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(70000.0)), f32::INFINITY);
    }
}
