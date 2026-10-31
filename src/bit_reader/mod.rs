#![allow(non_snake_case)]
use core::default::Default;
macro_rules! xprintln (
  ($a : expr) => ();
  ($a : expr, $b : expr) => ();
  ($a : expr, $b : expr, $c : expr) => ();
  ($a : expr, $b : expr, $c : expr, $d : expr) => ();
  ($a : expr, $b : expr, $c : expr, $d : expr, $e : expr) => ();
  ($a : expr, $b : expr, $c : expr, $d : expr, $e : expr, $f : expr) => ();
);

#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub type reg_t = u64;

#[cfg(not(target_pointer_width = "64"))]
#[allow(non_camel_case_types)]
pub type reg_t = u32;

pub const BROTLI_SHORT_FILL_BIT_WINDOW_READ: u32 = (::core::mem::size_of::<reg_t>() >> 1) as u32;

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
#[allow(non_upper_case_globals)]
const kBitMask: [u32; 33] =
  [0x0000, 0x00000001, 0x00000003, 0x00000007, 0x0000000F, 0x0000001F, 0x0000003F, 0x0000007F,
   0x000000FF, 0x000001FF, 0x000003FF, 0x000007FF, 0x00000FFF, 0x00001FFF, 0x00003FFF, 0x00007FFF,
   0x0000FFFF, 0x0001FFFF, 0x0003FFFF, 0x0007FFFF, 0x000FFFFF, 0x001FFFFF, 0x003FFFFF, 0x007FFFFF,
   0x00FFFFFF, 0x01FFFFFF, 0x03FFFFFF, 0x07FFFFFF, 0x0FFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF,
   0xFFFFFFFF];

// Equals kBitMask[n] without the table load and its bounds check; C computes
// it the same way wherever it has a bit-field extract instruction.
// On 64-bit targets it is computed in 64 bits, so it holds for every n in
// 0..=32. On 32-bit targets a 64-bit shift is a multi-instruction sequence,
// and making n == 32 work costs the hot paths an extra compare and select
// (+1.4% instructions on Cortex-A53), so it holds for n in 0..=31 only: the
// 32-bit reader never needs a 32-bit mask, as it splits reads wider than 16
// bits into two (`BrotliReadBits`, `BrotliSafeReadBits32Slow`) and peeks at
// most 16 bits.
#[inline(always)]
pub fn BitMask(n: u32) -> u32 {
  #[cfg(target_pointer_width = "64")]
  {
    (!(u64::MAX << (n & 63))) as u32
  }
  #[cfg(not(target_pointer_width = "64"))]
  {
    debug_assert!(n < 32, "BitMask({}) on a 32-bit target", n);
    !(u32::MAX << (n & 31))
  }
}

pub struct BrotliBitReader {
  pub val_: reg_t, // pre-fetched bits
  pub bit_pos_: u32, // current bit-reading position in val_
  pub next_in: u32, // the byte we're reading from
  pub avail_in: u32,
}

impl Default for BrotliBitReader {
  fn default() -> Self {
    BrotliBitReader {
      val_: 0,
      bit_pos_: 0,
      next_in: 0,
      avail_in: 0,
    }
  }
}

pub struct BrotliBitReaderState {
  pub val_: reg_t, // pre-fetched bits
  pub bit_pos_: u32, // current bit-reading position in val_
  pub next_in: u32, // the byte we're reading from
  pub avail_in: u32,
}
impl Default for BrotliBitReaderState {
  #[inline]
  fn default() -> Self {
    BrotliBitReaderState {
      val_: 0,
      bit_pos_: 0,
      next_in: 0,
      avail_in: 0,
    }
  }
}

pub fn BrotliBitReaderSaveState(from: &BrotliBitReader) -> BrotliBitReaderState {
  BrotliBitReaderState {
    val_: from.val_,
    bit_pos_: from.bit_pos_,
    next_in: from.next_in,
    avail_in: from.avail_in,
  }
}

pub fn BrotliBitReaderRestoreState(to: &mut BrotliBitReader, from: &BrotliBitReaderState) {
  to.val_ = from.val_;
  to.bit_pos_ = from.bit_pos_;
  to.next_in = from.next_in;
  to.avail_in = from.avail_in;
}

pub fn BrotliGetAvailableBits(br: &BrotliBitReader) -> u32 {
  ((::core::mem::size_of::<reg_t>() as u32) << 3) - br.bit_pos_
}

// The bit reader keeps its cursor in u32, so the one-past-the-end offset must
// fit too: bounding the offset and the length separately would still let
// consuming the last byte overflow next_in. usize may be 64 bits wide, so the
// sum saturates rather than wraps.
#[inline]
pub fn is_valid_input_range(input_offset: usize, available_in: usize, input_len: usize) -> bool {
  let end = (input_offset as u64).saturating_add(available_in as u64);
  end <= input_len as u64 && end <= u64::from(u32::MAX)
}

pub fn is_valid_bit_reader(br: &BrotliBitReader, input: &[u8]) -> bool {
  let register_bits = (::core::mem::size_of::<reg_t>() as u32) << 3;
  br.bit_pos_ <= register_bits &&
    u64::from(br.next_in) + u64::from(br.avail_in) <= input.len() as u64
}

// Returns amount of unread bytes the bit reader still has buffered from the
// BrotliInput, including whole bytes in br->val_.
pub fn BrotliGetRemainingBytes(br: &BrotliBitReader) -> u32 {
  br.avail_in + (BrotliGetAvailableBits(br) >> 3)
}

// Checks if there is at least num bytes left in the input ringbuffer (excluding
// the bits remaining in br->val_).
pub fn BrotliCheckInputAmount(br: &BrotliBitReader, num: u32) -> bool {
  br.avail_in >= num
}


#[inline(always)]
// Callers refill only when the caller checked enough input is available, so
// the zero fallback is unreachable; it replaces a panicking index.
fn BrotliLoad16LE(input: &[u8], next_in_u32: u32) -> u16 {
  match input.get(next_in_u32 as usize..) {
    Some(&[a, b, ..]) => u16::from_le_bytes([a, b]),
    _ => {
      debug_assert!(false, "BrotliLoad16LE past the end of the input");
      0
    }
  }
}


// from_le_bytes compiles to one unaligned load; assembling the value byte by
// byte let LLVM split the load into pieces once callers shift out high bytes.
#[allow(dead_code)]
#[inline(always)]
fn BrotliLoad32LE(input: &[u8], next_in_u32: u32) -> u32 {
  let next_in: usize = next_in_u32 as usize;
  let mut four_byte: [u8; 4] = fast_uninitialized![4];
  four_byte.clone_from_slice(fast!((input)[next_in ; next_in + 4]));
  u32::from_le_bytes(four_byte)
}

#[allow(dead_code)]
#[inline(always)]
fn BrotliLoad64LE(input: &[u8], next_in_u32: u32) -> u64 {
  let next_in: usize = next_in_u32 as usize;
  let mut eight_byte: [u8; 8] = fast_uninitialized![8];
  eight_byte.clone_from_slice(fast!((input)[next_in ; next_in + 8]));
  u64::from_le_bytes(eight_byte)
}
#[allow(dead_code)]
pub const BROTLI_ALIGNED_READ: u8 = 0;

#[allow(dead_code)]
#[inline(always)]
pub fn BrotliFillBitWindow(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) {
  BrotliFillBitWindowCompileTimeNbits(br, n_bits, input);
}

#[inline(always)]
fn BrotliFillBitWindowCompileTimeNbits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) {
  // The 32-bit register always refills 16 bits, whatever `n_bits` is.
  #[cfg(not(target_pointer_width = "64"))]
  let _ = n_bits;
  if ::core::mem::size_of::<reg_t>() == 8 {
    #[cfg(target_pointer_width = "64")]
    {
      if BROTLI_ALIGNED_READ == 0 && n_bits <= 8 {
        // !BROTLI_ALIGNED_READ && IS_CONSTANT(n_bits) && (n_bits <= 8)) {
        if br.bit_pos_ >= 56 {
          br.val_ >>= 56;
          br.bit_pos_ ^= 56;  /* here same as -= 56 because of the if condition */
          br.val_ |= BrotliLoad64LE(input, br.next_in) << 8;
          br.avail_in -= 7;
          br.next_in += 7;
        }
      } else if BROTLI_ALIGNED_READ == 0 && n_bits <= 16 {
        // (!BROTLI_ALIGNED_READ && IS_CONSTANT(n_bits) && (n_bits <= 16)) {
        if br.bit_pos_ >= 48 {
          br.val_ >>= 48;
          br.bit_pos_ ^= 48;  /* here same as -= 48 because of the if condition */
          br.val_ |= BrotliLoad64LE(input, br.next_in) << 16;
          br.avail_in -= 6;
          br.next_in += 6;
        }
      } else if br.bit_pos_ >= 32 {
        br.val_ >>= 32;
        br.bit_pos_ ^= 32;  /* here same as -= 32 because of the if condition */
        br.val_ |= (BrotliLoad32LE(input, br.next_in) as reg_t) << 32;
        br.avail_in -= BROTLI_SHORT_FILL_BIT_WINDOW_READ;
        br.next_in += BROTLI_SHORT_FILL_BIT_WINDOW_READ;
      }
    }
  } else if br.bit_pos_ >= 16 {
      let next = BrotliLoad16LE(input, br.next_in) as reg_t;
      br.val_ = (br.val_ >> 16) | (next << 16);
      br.bit_pos_ &= 15;  /* here same as -= 16 because 16 <= br.bit_pos_ < 32 */
      br.avail_in -= BROTLI_SHORT_FILL_BIT_WINDOW_READ;
      br.next_in += BROTLI_SHORT_FILL_BIT_WINDOW_READ;
  }
}

// Mosltly like BrotliFillBitWindow, but guarantees only 16 bits and reads no
// more than BROTLI_SHORT_FILL_BIT_WINDOW_READ bytes of input.
pub fn BrotliFillBitWindow16(br: &mut BrotliBitReader, input: &[u8]) {
  BrotliFillBitWindowCompileTimeNbits(br, 17, input);
}

// Pulls one byte of input to accumulator.
pub fn BrotliPullByte(br: &mut BrotliBitReader, input: &[u8]) -> bool {
  if br.avail_in == 0 {
    return false;
  }
  br.val_ >>= 8;
  br.val_ |= (fast!((input)[br.next_in as usize]) as reg_t) << ((::core::mem::size_of::<reg_t>() << 3) - 8);
  br.bit_pos_ -= 8;
  br.avail_in -= 1;
  br.next_in += 1;
  true
}

// Returns currently available bits.
// The number of valid bits could be calclulated by BrotliGetAvailableBits.
#[inline(always)]
pub fn BrotliGetBitsUnmasked(br: &BrotliBitReader) -> reg_t {
  br.val_ >> br.bit_pos_
}

// Like BrotliGetBits, but does not mask the result.
// The result contains at least 16 valid bits.
#[inline(always)]
pub fn BrotliGet16BitsUnmasked(br: &mut BrotliBitReader, input: &[u8]) -> u32 {
  BrotliFillBitWindowCompileTimeNbits(br, 16, input);
  (BrotliGetBitsUnmasked(br) & (0xffffffffu32 as reg_t)) as u32
}

// Returns the specified number of bits from br without advancing bit pos.
#[allow(dead_code)]
#[inline(always)]
pub fn BrotliGetBits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) -> u32 {
  BrotliFillBitWindow(br, n_bits, input);
  (BrotliGetBitsUnmasked(br) as u32) & BitMask(n_bits)
}

// Returns the specified number of bits from br without advancing bit pos.
#[allow (dead_code)]
pub fn BrotliGetConstantNBits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) -> u32 {
  BrotliFillBitWindowCompileTimeNbits(br, n_bits, input);
  (BrotliGetBitsUnmasked(br) as u32) & BitMask(n_bits)
}

// Tries to peek the specified amount of bits. Returns 0, if there is not
// enough input.
pub fn BrotliSafeGetBits(br: &mut BrotliBitReader,
                         n_bits: u32,
                         val: &mut u32,
                         input: &[u8])
                         -> bool {
  while BrotliGetAvailableBits(br) < n_bits {
    if !BrotliPullByte(br, input) {
      return false;
    }
  }
  *val = (BrotliGetBitsUnmasked(br) as u32) & BitMask(n_bits);
  true
}

// Advances the bit pos by n_bits.
#[inline(always)]
pub fn BrotliDropBits(br: &mut BrotliBitReader, n_bits: u32) {
  br.bit_pos_ += n_bits;
}

pub fn BrotliBitReaderUnload(br: &mut BrotliBitReader) {
  let unused_bytes: u32 = BrotliGetAvailableBits(br) >> 3;
  let unused_bits: u32 = unused_bytes << 3;
  br.avail_in += unused_bytes;
  br.next_in -= unused_bytes;
  if unused_bits as usize == (::core::mem::size_of::<reg_t>() << 3) {
    br.val_ = 0;
  } else {
    br.val_ <<= unused_bits;
  }
  br.bit_pos_ += unused_bits;
}

// Reads the specified number of bits from br and advances the bit pos.
// Precondition: accumulator MUST contain at least n_bits.
#[inline(always)]
pub fn BrotliTakeBits(br: &mut BrotliBitReader, n_bits: u32, val: &mut u32) {
  *val = (BrotliGetBitsUnmasked(br) as u32) & BitMask(n_bits);
  // if true {
  xprintln!("[BrotliReadBits]  {:?} {:?} {:?} val: {:x}\n",
           br.avail_in, br.bit_pos_, n_bits, *val);
  // }
  BrotliDropBits(br, n_bits);
}

// Reads the specified number of bits from br and advances the bit pos.
// Assumes that there is enough input to perform BrotliFillBitWindow.
//
// Callers pass a data-dependent n_bits (extra bits of lengths and distances),
// for which C's BrotliReadBits24 compiles BrotliFillBitWindow to its generic
// case alone: refill 32 bits once half the window is consumed, which leaves at
// least 33 bits for any n_bits up to 32. Choosing among the narrower n_bits
// cases at run time, as before, costs extra unpredictable branches.
#[inline(always)]
pub fn BrotliReadBits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) -> u32 {
  if ::core::mem::size_of::<reg_t>() == 8 {
    let mut val: u32 = 0;
    BrotliFillBitWindowCompileTimeNbits(br, 32, input);
    BrotliTakeBits(br, n_bits, &mut val);
    val
  } else if n_bits <= 16 {
    let mut val: u32 = 0;
    BrotliFillBitWindowCompileTimeNbits(br, 16, input);
    BrotliTakeBits(br, n_bits, &mut val);
    val
  } else {
    let mut low_val: u32 = 0;
    let mut high_val: u32 = 0;
    BrotliFillBitWindowCompileTimeNbits(br, 16, input);
    BrotliTakeBits(br, 16, &mut low_val);
    BrotliFillBitWindowCompileTimeNbits(br, 16, input);
    BrotliTakeBits(br, n_bits - 16, &mut high_val);
    low_val | (high_val << 16)
  }
}

// Reads the specified number of bits from br and advances the bit pos.
// Assumes that there is enough input to perform BrotliFillBitWindow.
#[allow (dead_code)]
pub fn BrotliReadConstantNBits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) -> u32 {
  if ::core::mem::size_of::<reg_t>() == 8 || (n_bits <= 16) {
    let mut val: u32 = 0;
    BrotliFillBitWindowCompileTimeNbits(br, n_bits, input);
    BrotliTakeBits(br, n_bits, &mut val);
    val
  } else {
    let mut low_val: u32 = 0;
    let mut high_val: u32 = 0;
    BrotliFillBitWindowCompileTimeNbits(br, 16, input);
    BrotliTakeBits(br, 16, &mut low_val);
    BrotliFillBitWindowCompileTimeNbits(br, 8, input);
    BrotliTakeBits(br, n_bits - 16, &mut high_val);
    low_val | (high_val << 16)
  }
}

#[cold]
#[inline(never)]
fn BrotliSafeReadBits32Slow(br: &mut BrotliBitReader,
                            n_bits: u32,
                            val: &mut u32,
                            input: &[u8])
                            -> bool {
  let mut low_val: u32 = 0;
  let mut high_val: u32 = 0;
  let memento = BrotliBitReaderSaveState(br);
  if !BrotliSafeReadBits(br, 16, &mut low_val, input) ||
     !BrotliSafeReadBits(br, n_bits - 16, &mut high_val, input) {
    BrotliBitReaderRestoreState(br, &memento);
    return false;
  }
  *val = low_val | (high_val << 16);
  true
}

// Tries to read the specified amount of bits. Returns 0, if there is not
// enough input. n_bits MUST be positive.
pub fn BrotliSafeReadBits(br: &mut BrotliBitReader,
                          n_bits: u32,
                          val: &mut u32,
                          input: &[u8])
                          -> bool {
  if ::core::mem::size_of::<reg_t>() < 8 && n_bits > 24 {
    return BrotliSafeReadBits32Slow(br, n_bits, val, input);
  }
  while BrotliGetAvailableBits(br) < n_bits {
    if !BrotliPullByte(br, input) {
      return false;
    }
  }
  BrotliTakeBits(br, n_bits, val);
  true
}

// Advances the bit reader position to the next byte boundary and verifies
// that any skipped bits are set to zero.
pub fn BrotliJumpToByteBoundary(br: &mut BrotliBitReader) -> bool {
  let pad_bits_count: u32 = BrotliGetAvailableBits(br) & 0x7;
  let mut pad_bits: u32 = 0;
  if pad_bits_count != 0 {
    BrotliTakeBits(br, pad_bits_count, &mut pad_bits);
  }
  pad_bits == 0
}

// Peeks a byte at specified offset.
// Precondition: bit reader is parked to a byte boundary.
// Returns -1 if operation is not feasible.
pub fn BrotliPeekByte(br: &mut BrotliBitReader, mut offset: u32, input: &[u8]) -> i32 {
  if !is_valid_bit_reader(br, input) {
    return -1;
  }
  let available_bits: u32 = BrotliGetAvailableBits(br);
  let bytes_left: u32 = (available_bits >> 3);
  if (available_bits & 7) != 0 {
    return -1;
  }
  if offset < bytes_left {
    return ((BrotliGetBitsUnmasked(br) >> ((offset << 3)) as u32) & 0xFF) as i32;
  }
  offset -= bytes_left;
  if offset < br.avail_in {
    let input_index = u64::from(br.next_in) + u64::from(offset);
    return match input.get(input_index as usize) {
      Some(value) => *value as i32,
      None => -1,
    };
  }
  -1
}

// Copies remaining input bytes stored in the bit reader to the output. Value
// num may not be larger than BrotliGetRemainingBytes. The bit reader must be
// warmed up again after this.
pub fn BrotliCopyBytes(dest: &mut [u8], br: &mut BrotliBitReader, mut num: u32, input: &[u8]) {
  let mut offset: u32 = 0;
  while BrotliGetAvailableBits(br) >= 8 && num > 0 {
    fast_mut!((dest)[offset as usize]) = BrotliGetBitsUnmasked(br) as u8;
    BrotliDropBits(br, 8);
    offset += 1;
    num -= 1;
  }
  // One bulk copy of the same bytes the per-byte loop used to copy.
  let dest_start = offset as usize;
  let input_start = br.next_in as usize;
  fast_mut!((dest)[dest_start ; dest_start + num as usize])
    .clone_from_slice(fast!((input)[input_start ; input_start + num as usize]));
  br.avail_in -= num;
  br.next_in += num;
}

pub fn BrotliInitBitReader(br: &mut BrotliBitReader) {
  br.val_ = 0;
  br.bit_pos_ = (::core::mem::size_of::<reg_t>() << 3) as u32;
}

pub fn BrotliWarmupBitReader(br: &mut BrotliBitReader, input: &[u8]) -> bool {
  let _aligned_read_mask: usize = (::core::mem::size_of::<reg_t>() >> 1) - 1;
  // Fixing alignment after unaligned BrotliFillWindow would result accumulator
  // overflow. If unalignment is caused by BrotliSafeReadBits, then there is
  // enough space in accumulator to fix aligment.
  if BrotliGetAvailableBits(br) == 0 && !BrotliPullByte(br, input) {
    return false;
  }
  // we can't tell about alignment yet
  // while (((size_t)br->next_in) & aligned_read_mask) != 0 {
  // if !BrotliPullByte(br) {
  // If we consumed all the input, we don't care about the alignment. */
  // return true;
  // }
  // }
  //
  true
}


// A short-lived copy of a `BrotliBitReader` for the decoder's hot loops.
//
// The unread input is held as a subslice `tail` of the input, so the slice
// length is the number of bytes left (`avail_in`) and a refill is one length
// check, one load and one pointer bump, like the C decoder's `next_in`
// pointer. `BrotliBitReader` instead keeps `next_in` and `avail_in`, so every
// refill bounds-checks `next_in` against `input.len()` and also updates
// `avail_in`; on 32-bit targets, which refill every 16 bits, that is a large
// part of the cost of a command. Bits are consumed exactly as by the
// `BrotliBitReader` functions with the same `n_bits`. Callers must have
// checked that enough input is available (as the non-safe decoding paths
// do); if not, missing bytes read as zero instead of panicking.
pub struct BrotliFastBitReader<'a> {
  pub val_: reg_t,
  pub bit_pos_: u32,
  tail: &'a [u8],
  start_len: usize,
}

impl<'a> BrotliFastBitReader<'a> {
  #[inline(always)]
  pub fn new(br: &BrotliBitReader, input: &'a [u8]) -> Self {
    let start = br.next_in as usize;
    let tail = match input.get(start..) {
      Some(rest) => rest.get(..br.avail_in as usize).unwrap_or(rest),
      None => &[],
    };
    BrotliFastBitReader {
      val_: br.val_,
      bit_pos_: br.bit_pos_,
      tail: tail,
      start_len: tail.len(),
    }
  }

  // Writes the state back to `br`, which must be the reader it was created
  // from.
  #[inline(always)]
  pub fn store(&self, br: &mut BrotliBitReader) {
    let consumed = (self.start_len - self.tail.len()) as u32;
    br.val_ = self.val_;
    br.bit_pos_ = self.bit_pos_;
    br.next_in = br.next_in.wrapping_add(consumed);
    br.avail_in = br.avail_in.wrapping_sub(consumed);
  }

  // Bytes of input not yet loaded into the register; the same as
  // `BrotliBitReader::avail_in`.
  #[inline(always)]
  pub fn avail_in(&self) -> u32 {
    self.tail.len() as u32
  }

  // Same as `BrotliFillBitWindowCompileTimeNbits`.
  #[inline(always)]
  pub fn fill(&mut self, n_bits: u32) {
    #[cfg(target_pointer_width = "64")]
    {
      if n_bits <= 8 {
        if self.bit_pos_ >= 56 {
          self.val_ >>= 56;
          self.bit_pos_ ^= 56;
          if let &[a, b, c, d, e, f, g, h, ..] = self.tail {
            self.tail = &self.tail[7..];
            self.val_ |= u64::from_le_bytes([a, b, c, d, e, f, g, h]) << 8;
          }
        }
      } else if n_bits <= 16 {
        if self.bit_pos_ >= 48 {
          self.val_ >>= 48;
          self.bit_pos_ ^= 48;
          if let &[a, b, c, d, e, f, g, h, ..] = self.tail {
            self.tail = &self.tail[6..];
            self.val_ |= u64::from_le_bytes([a, b, c, d, e, f, g, h]) << 16;
          }
        }
      } else if self.bit_pos_ >= 32 {
        self.val_ >>= 32;
        self.bit_pos_ ^= 32;
        if let &[a, b, c, d, ref rest @ ..] = self.tail {
          self.tail = rest;
          self.val_ |= (u32::from_le_bytes([a, b, c, d]) as reg_t) << 32;
        }
      }
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
      let _ = n_bits;
      if self.bit_pos_ >= 16 {
        self.val_ >>= 16;
        self.bit_pos_ &= 15;
        if let &[a, b, ref rest @ ..] = self.tail {
          self.tail = rest;
          self.val_ |= (u16::from_le_bytes([a, b]) as reg_t) << 16;
        }
      }
    }
  }

  #[inline(always)]
  pub fn get_bits_unmasked(&self) -> reg_t {
    self.val_ >> self.bit_pos_
  }

  // Same as `BrotliGet16BitsUnmasked`.
  #[inline(always)]
  pub fn get_16_bits_unmasked(&mut self) -> u32 {
    self.fill(16);
    (self.get_bits_unmasked() & (0xffffffffu32 as reg_t)) as u32
  }

  // Same as `BrotliGetBits`.
  #[inline(always)]
  pub fn get_bits(&mut self, n_bits: u32) -> u32 {
    self.fill(n_bits);
    (self.get_bits_unmasked() as u32) & BitMask(n_bits)
  }

  #[inline(always)]
  pub fn drop_bits(&mut self, n_bits: u32) {
    self.bit_pos_ += n_bits;
  }

  #[inline(always)]
  fn take_bits(&mut self, n_bits: u32) -> u32 {
    let val = (self.get_bits_unmasked() as u32) & BitMask(n_bits);
    self.drop_bits(n_bits);
    val
  }

  // Same as `BrotliReadBits`.
  #[inline(always)]
  pub fn read_bits(&mut self, n_bits: u32) -> u32 {
    if ::core::mem::size_of::<reg_t>() == 8 {
      self.fill(32);
      self.take_bits(n_bits)
    } else {
      self.fill(16);
      if n_bits <= 16 {
        self.take_bits(n_bits)
      } else {
        let low_val = self.get_bits_unmasked() as u16 as u32;
        self.val_ >>= 16;
        if let &[a, b, ref rest @ ..] = self.tail {
          self.tail = rest;
          self.val_ |= (u16::from_le_bytes([a, b]) as reg_t) << 16;
        }
        let high_val = self.take_bits(n_bits - 16);
        low_val | (high_val << 16)
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bit_mask_matches_the_table() {
    // 32-bit targets support n <= 31 only; see `BitMask`.
    let max = if ::core::mem::size_of::<reg_t>() == 8 { 32 } else { 31 };
    for n in 0..max + 1 {
      assert_eq!(BitMask(n), kBitMask[n as usize]);
    }
  }

  #[test]
  fn input_range_checks_the_end_cursor() {
    // Exercise the range arithmetic without allocating a multi-GiB input.
    let max_cursor = u32::MAX as usize;
    assert!(is_valid_input_range(0, 0, 0));
    assert!(is_valid_input_range(3, 5, 8));
    assert!(!is_valid_input_range(3, 6, 8));
    assert!(!is_valid_input_range(9, 0, 8));
    assert!(is_valid_input_range(0, max_cursor, max_cursor));
    assert!(is_valid_input_range(max_cursor - 1, 1, max_cursor));
    assert!(is_valid_input_range(max_cursor, 0, max_cursor));
    assert!(!is_valid_input_range(max_cursor, 1, usize::MAX));
    assert!(!is_valid_input_range(max_cursor - 1, 2, usize::MAX));
    assert!(!is_valid_input_range(usize::MAX, 1, usize::MAX));
  }

  #[cfg(target_pointer_width="64")]
  #[test]
  fn a_large_input_slice_does_not_widen_the_cursor() {
    let input_len = 1usize << 32;
    assert!(is_valid_input_range(0, input_len - 1, input_len));
    assert!(!is_valid_input_range(input_len - 1, 1, input_len));
    assert!(!is_valid_input_range(1, input_len - 1, input_len));
    assert!(!is_valid_input_range(0, input_len, input_len));
    assert!(!is_valid_input_range(input_len, 0, input_len));
  }

  #[test]
  fn peek_byte_rejects_invalid_state() {
    let data: [u8; 1] = [0x7f];
    let mut unaligned_reader = BrotliBitReader {
      val_: 0,
      bit_pos_: 1,
      avail_in: 0,
      next_in: 0,
    };
    assert_eq!(BrotliPeekByte(&mut unaligned_reader, 0, &data), -1);

    let mut out_of_bounds_reader = BrotliBitReader {
      val_: 0,
      bit_pos_: 64,
      avail_in: 1,
      next_in: 1,
    };
    assert_eq!(BrotliPeekByte(&mut out_of_bounds_reader, 0, &data), -1);
  }

  // Hard-coded vectors for the 64-bit register; regression_tests.rs
  // checks both register widths against a bit-at-a-time oracle.
  #[cfg(target_pointer_width = "64")]
  #[test]
  fn warmup_works() {
    let data: [u8; 37] = [0xde, 0xad, 0xbe, 0xef /* 4 bytes garbage at beginning */, 0x21,
                          0xfd, 0xff, 0x87, 0x03, 0x6c, 0x63, 0x90, 0xde, 0xe9, 0x36, 0x04, 0xf8,
                          0x89, 0xb4, 0x4d, 0x57, 0x96, 0x51, 0x0a, 0x54, 0xb7, 0x54, 0xee, 0xf4,
                          0xdc, 0xa0, 0x05, 0x9b, 0xd2, 0x2b, 0x30, 0x8f]; // 33 b goods
    let mut bit_reader = BrotliBitReader {
      val_: 0x0,
      bit_pos_: 64,
      avail_in: 33,
      next_in: 4,
    };
    let ret = BrotliWarmupBitReader(&mut bit_reader, &data[..]);
    assert_eq!(ret, true);
    assert_eq!(bit_reader.val_, 0x2100000000000000);
    assert_eq!(bit_reader.bit_pos_, 56);
    assert_eq!(bit_reader.avail_in, 32);
    assert_eq!(bit_reader.next_in, 5);
  }
  // Hard-coded vectors for the 64-bit register; regression_tests.rs
  // checks both register widths against a bit-at-a-time oracle.
  #[cfg(target_pointer_width = "64")]
  #[test]
  fn warmup_errcheck() {
    let data: [u8; 3] = [0xde, 0xad, 0xeb]; // 3 bytes garbage at beginning
    let mut bit_reader = BrotliBitReader {
      val_: 0x86e884e1ffff577b,
      bit_pos_: 64,
      avail_in: 0,
      next_in: 3,
    };
    let ret = BrotliWarmupBitReader(&mut bit_reader, &data[..]);
    assert_eq!(ret, false);
    assert_eq!(bit_reader.val_, 0x86e884e1ffff577b);
    assert_eq!(bit_reader.bit_pos_, 64);
    assert_eq!(bit_reader.avail_in, 0);
    assert_eq!(bit_reader.next_in, 3);
  }
  // Hard-coded vectors for the 64-bit register; regression_tests.rs
  // checks both register widths against a bit-at-a-time oracle.
  #[cfg(target_pointer_width = "64")]
  #[test]
  fn safe_read_tests() {
    {
      let data: [u8; 12] = [0x50, 0x3b, 0xbb, 0x5e, 0xc5, 0x96, 0x81, 0xb7, 0x52, 0x89, 0xea, 0x3d];
      let mut bit_reader = BrotliBitReader {
        val_: 0xe1e56a736e04fbf5,
        bit_pos_: 6,
        avail_in: 12,
        next_in: 0,
      };
      let mut val: u32 = 0;
      let ret = BrotliSafeReadBits(&mut bit_reader, 7, &mut val, &data[..]);
      assert_eq!(ret, true);
      assert_eq!(val, 0x6f);
      assert_eq!(bit_reader.avail_in, 12);
      assert_eq!(bit_reader.next_in, 0);
      assert_eq!(bit_reader.bit_pos_, 13);
    }
    {
      let data: [u8; 7] = [0xba, 0xd0, 0xf0, 0x0d, 0xc8, 0xcd, 0xcc];
      let mut bit_reader = BrotliBitReader {
        val_: 0x17f115ae26916f0,
        bit_pos_: 57,
        avail_in: 3,
        next_in: 4,
      };
      let mut val: u32 = 0;
      let ret = BrotliSafeReadBits(&mut bit_reader, 15, &mut val, &data[..]);
      assert_eq!(ret, true);
      assert_eq!(val, 0x6400);
      assert_eq!(bit_reader.avail_in, 2);
      assert_eq!(bit_reader.next_in, 5);
      assert_eq!(bit_reader.bit_pos_, 64);
    }
    {
      let data: [u8; 4] = [0xee, 0xee, 0xf0, 0xd5];
      let mut bit_reader = BrotliBitReader {
        val_: 0x5f43f252c027e447,
        bit_pos_: 53,
        avail_in: 2,
        next_in: 2,
      };
      let mut val: u32 = 0;
      let ret = BrotliSafeReadBits(&mut bit_reader, 14, &mut val, &data[..]);
      assert_eq!(ret, true);
      assert_eq!(bit_reader.avail_in, 1);
      assert_eq!(bit_reader.next_in, 3);
      assert_eq!(bit_reader.bit_pos_, 59);
      assert_eq!(bit_reader.val_, 0xf05f43f252c027e4);
      assert_eq!(val, 0x2fa);
    }
    {
      let data: [u8; 4] = [0xee, 0xee, 0xf0, 0xd5];
      let mut bit_reader = BrotliBitReader {
        val_: 0x2f902339697460,
        bit_pos_: 57,
        avail_in: 0,
        next_in: 4,
      };
      let mut val: u32 = 0x74eca3f0;
      let ret = BrotliSafeReadBits(&mut bit_reader, 14, &mut val, &data[..]);
      assert_eq!(ret, false);
      assert_eq!(bit_reader.avail_in, 0);
      assert_eq!(bit_reader.next_in, 4);
      assert_eq!(bit_reader.bit_pos_, 57);
      assert_eq!(bit_reader.val_, 0x2f902339697460);
      assert_eq!(val, 0x74eca3f0);
    }
  }
  // Hard-coded vectors for the 64-bit register; regression_tests.rs
  // checks both register widths against a bit-at-a-time oracle.
  #[cfg(target_pointer_width = "64")]
  #[test]
  fn bit_read_tests() {
    {
      let data: [u8; 32] = [0xba, 0xaa, 0xad, 0xdd, 0x57, 0x5c, 0xd9, 0xa3, 0x3e, 0xb3, 0x77,
                            0xe7, 0xa0, 0x1e, 0x09, 0xd3, 0x12, 0xa1, 0x3f, 0xb8, 0x7e, 0x5a,
                            0x06, 0x86, 0xe5, 0x36, 0xef, 0x9c, 0x9f, 0x6d, 0x9b, 0xcc];
      let mut bit_reader = BrotliBitReader {
        val_: 0xf5917f07daaaeabb,
        bit_pos_: 33,
        avail_in: 29,
        next_in: 3,
      };
      let ret = BrotliReadBits(&mut bit_reader, 8, &data[..]);
      assert_eq!(ret, 0x83);
      assert_eq!(bit_reader.bit_pos_, 9);
      assert_eq!(bit_reader.avail_in, 25);
      assert_eq!(bit_reader.next_in, 7);
    }
    {
      let data: [u8; 28] = [0xba, 0xaa, 0xaa, 0xad, 0x74, 0x40, 0x8e, 0xee, 0xd2, 0x38, 0xf1,
                            0xf4, 0xf8, 0x1d, 0x9f, 0x24, 0x48, 0x1e, 0x82, 0xce, 0x48, 0x88,
                            0xd7, 0x25, 0x74, 0xaf, 0xe3, 0xea];
      let mut bit_reader = BrotliBitReader {
        val_: 0x27e33b2440d3feaf,
        bit_pos_: 18,
        avail_in: 24,
        next_in: 4,
      };
      let ret = BrotliReadBits(&mut bit_reader, 15, &data[..]);
      assert_eq!(ret, 0x1034);
      assert_eq!(bit_reader.bit_pos_, 33);
      assert_eq!(bit_reader.avail_in, 24);
      assert_eq!(bit_reader.next_in, 4);
    }
  }
  // Hard-coded vectors for the 64-bit register; regression_tests.rs
  // checks both register widths against a bit-at-a-time oracle.
  #[cfg(target_pointer_width = "64")]
  #[test]
  fn const_fill_bit_window_tests() {
    {
      let data: [u8; 36] = [0xff, 0x9a, 0xa0, 0xde, 0x50, 0x99, 0x67, 0x67, 0x69, 0x87, 0x0e,
                            0x69, 0xeb, 0x6a, 0xd1, 0x56, 0xc0, 0x32, 0x96, 0xed, 0x78, 0x0e,
                            0x19, 0xdd, 0x0b, 0xe8, 0xf8, 0x33, 0x9f, 0xe0, 0x69, 0x55, 0x59,
                            0x3f, 0x5d, 0xc8];
      let mut bit_reader = BrotliBitReader {
        val_: 0xb3fc441e0181dc4,
        bit_pos_: 59,
        avail_in: 33,
        next_in: 1,
      };
      let ret = BrotliReadConstantNBits(&mut bit_reader, 4, &data[..]);
      assert_eq!(ret, 0x1);
      assert_eq!(bit_reader.bit_pos_, 7);
      assert_eq!(bit_reader.avail_in, 26);
      assert_eq!(bit_reader.next_in, 8);
    }
    {
      let data: [u8; 67] = [0xf0, 0x0d, 0x7e, 0x18, 0x70, 0x1c, 0x18, 0x57, 0xbd, 0x73, 0x47,
                            0xc1, 0xb4, 0xf7, 0xe2, 0xbe, 0x17, 0x6e, 0x26, 0x01, 0xb2, 0xd5,
                            0x55, 0xd8, 0x68, 0x1b, 0xc2, 0x87, 0xb4, 0xb1, 0xd9, 0x42, 0xac,
                            0x0d, 0x67, 0xb1, 0x93, 0x54, 0x49, 0xa4, 0x69, 0xf8, 0x16, 0x0e,
                            0x61, 0xb3, 0xdb, 0x98, 0xbb, 0xeb, 0xfa, 0xcb, 0x14, 0xcd, 0x68,
                            0x77, 0xa1, 0x33, 0x6c, 0x49, 0xfa, 0x35, 0xbb, 0xeb, 0xee, 0x7b, 0xae];
      let mut bit_reader = BrotliBitReader {
        val_: 0x655b1fe0dd6f1e78,
        bit_pos_: 63,
        avail_in: 65,
        next_in: 2,
      };
      let ret = BrotliGet16BitsUnmasked(&mut bit_reader, &data[..]);
      assert_eq!(ret, 0x38e030fc);
      assert_eq!(bit_reader.bit_pos_, 15);
      assert_eq!(bit_reader.avail_in, 59);
      assert_eq!(bit_reader.next_in, 8);
    }
  }
}
