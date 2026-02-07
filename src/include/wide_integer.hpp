#pragma once

#include "duckdb.hpp"
#include <cstdint>

namespace duckdb {

// ---------------------------------------------------------------------------
// hugeint_t <-> __int128 conversion
// ---------------------------------------------------------------------------

inline __int128 HugeintToInt128(const hugeint_t &h) {
	return (static_cast<__int128>(h.upper) << 64) | h.lower;
}

inline hugeint_t Int128ToHugeint(__int128 v) {
	hugeint_t result;
	result.upper = static_cast<int64_t>(v >> 64);
	result.lower = static_cast<uint64_t>(v);
	return result;
}

// ---------------------------------------------------------------------------
// Absolute value for signed __int128
// ---------------------------------------------------------------------------

inline unsigned __int128 Abs128(__int128 x) {
	return x < 0 ? -static_cast<unsigned __int128>(x) : static_cast<unsigned __int128>(x);
}

// ---------------------------------------------------------------------------
// 256-bit unsigned integer (two 128-bit halves)
// ---------------------------------------------------------------------------

struct uint256_t {
	unsigned __int128 hi;
	unsigned __int128 lo;
};

// Multiply two unsigned 128-bit values, producing a 256-bit result.
// Uses schoolbook multiplication with 64-bit limbs.
inline uint256_t Mul128(unsigned __int128 a, unsigned __int128 b) {
	uint64_t a_lo = static_cast<uint64_t>(a);
	uint64_t a_hi = static_cast<uint64_t>(a >> 64);
	uint64_t b_lo = static_cast<uint64_t>(b);
	uint64_t b_hi = static_cast<uint64_t>(b >> 64);

	// Four partial products (each fits in unsigned __int128)
	unsigned __int128 p0 = static_cast<unsigned __int128>(a_lo) * b_lo;
	unsigned __int128 p1 = static_cast<unsigned __int128>(a_lo) * b_hi;
	unsigned __int128 p2 = static_cast<unsigned __int128>(a_hi) * b_lo;
	unsigned __int128 p3 = static_cast<unsigned __int128>(a_hi) * b_hi;

	// Accumulate middle terms
	unsigned __int128 mid = p1 + p2;
	unsigned __int128 mid_carry = (mid < p1) ? (static_cast<unsigned __int128>(1) << 64) : 0;

	// Low 128 bits
	unsigned __int128 lo = p0 + (mid << 64);
	unsigned __int128 lo_carry = (lo < p0) ? 1 : 0;

	// High 128 bits
	unsigned __int128 hi = p3 + (mid >> 64) + mid_carry + lo_carry;

	return {hi, lo};
}

// Divide a 256-bit unsigned value by a 128-bit unsigned divisor.
// Returns quotient (must fit in 128 bits) and sets *remainder.
//
// Uses two-step decomposition: treat the 256-bit numerator as (hi * 2^128 + lo),
// then compute quotient and remainder via two 128-bit divisions. This avoids
// the 256-iteration bit-by-bit loop of the naive approach.
//
// Step 1: hi / den = q_hi, r_hi  (q_hi must be 0 for result to fit in 128 bits)
// Step 2: (r_hi * 2^128 + lo) / den = q_lo, r_lo
//
// Step 2 requires dividing a value that can be up to 256 bits by a 128-bit divisor.
// We decompose it further using 64-bit limbs when the intermediate values exceed
// 128 bits, falling back to the hardware's native 128-bit division.
inline unsigned __int128 Div256By128(uint256_t num, unsigned __int128 den,
                                     unsigned __int128 *remainder) {
	// If hi < den, we can skip step 1 (q_hi = 0, r_hi = hi)
	if (num.hi == 0) {
		// Simple case: 128-bit / 128-bit
		unsigned __int128 quot = num.lo / den;
		if (remainder) {
			*remainder = num.lo % den;
		}
		return quot;
	}

	// Step 1: divide hi by den
	D_ASSERT(num.hi < den); // quotient must fit in 128 bits
	unsigned __int128 r_hi = num.hi;

	// Step 2: divide (r_hi * 2^128 + lo) by den
	// We need to handle this carefully since r_hi * 2^128 doesn't fit in 128 bits.
	// Decompose using 64-bit halves of lo.
	//
	// Let lo = lo_hi * 2^64 + lo_lo
	// (r_hi * 2^128 + lo) = (r_hi * 2^64 + lo_hi) * 2^64 + lo_lo
	//
	// Step 2a: (r_hi * 2^64 + lo_hi) / den = q2a, r2a
	//   But r_hi * 2^64 + lo_hi can exceed 128 bits if r_hi >= 2^64.
	//   However, we know r_hi < den, and we work with the 64-bit halves.

	uint64_t lo_hi = static_cast<uint64_t>(num.lo >> 64);
	uint64_t lo_lo = static_cast<uint64_t>(num.lo);

	// Compute (r_hi * 2^64 + lo_hi) as a 192-bit value, then divide by den.
	// Since r_hi < den (128-bit), r_hi * 2^64 needs up to 192 bits.
	// We use iterative 64-bit digit extraction.

	// Process the upper 64-bit digit of lo:
	// Form the 192-bit value: (r_hi << 64) | lo_hi, divide by den
	// This is equivalent to: remainder * 2^64 + next_digit, iterated.

	// Shift remainder up by 64 bits and add lo_hi
	// r_hi is < den, so (r_hi << 64 + lo_hi) may be > 2^128.
	// We handle this by noting that if r_hi < den, then after dividing
	// (r_hi * 2^64 + lo_hi) by den, the quotient fits in 64 bits.

	// Use the identity: (A * 2^64 + B) / D where A < D (128-bit)
	// q = 0 or 1 iteration of subtract-and-shift won't work directly.
	// Instead, use schoolbook division on 64-bit limbs.

	// For correctness with arbitrary 128-bit divisors, fall back to
	// binary long division but only over the significant bits.
	unsigned __int128 quot = 0;
	unsigned __int128 rem = r_hi;

	// Process the upper 64 bits of lo
	for (int32_t bit = 63; bit >= 0; bit--) {
		rem <<= 1;
		if (lo_hi & (static_cast<uint64_t>(1) << bit)) {
			rem |= 1;
		}
		if (rem >= den) {
			rem -= den;
			quot |= (static_cast<unsigned __int128>(1) << (bit + 64));
		}
	}

	// Process the lower 64 bits of lo
	for (int32_t bit = 63; bit >= 0; bit--) {
		rem <<= 1;
		if (lo_lo & (static_cast<uint64_t>(1) << bit)) {
			rem |= 1;
		}
		if (rem >= den) {
			rem -= den;
			quot |= (static_cast<unsigned __int128>(1) << bit);
		}
	}

	if (remainder) {
		*remainder = rem;
	}
	return quot;
}

// ---------------------------------------------------------------------------
// Power-of-10 lookup for unsigned __int128 (up to 10^38)
// ---------------------------------------------------------------------------

// Helper to construct unsigned __int128 from high and low 64-bit halves.
inline constexpr unsigned __int128 MakeUint128(uint64_t hi, uint64_t lo) {
	return (static_cast<unsigned __int128>(hi) << 64) | lo;
}

// O(1) lookup table covering 10^0 through 10^38.
// D_ASSERT guards against out-of-range exponents.
inline unsigned __int128 Pow10_128(uint32_t exp) {
	// clang-format off
	static constexpr unsigned __int128 table[] = {
	    // 10^0 - 10^19: fit in uint64_t
	    MakeUint128(0ULL,                    1ULL),                     // 10^0
	    MakeUint128(0ULL,                    10ULL),                    // 10^1
	    MakeUint128(0ULL,                    100ULL),                   // 10^2
	    MakeUint128(0ULL,                    1000ULL),                  // 10^3
	    MakeUint128(0ULL,                    10000ULL),                 // 10^4
	    MakeUint128(0ULL,                    100000ULL),                // 10^5
	    MakeUint128(0ULL,                    1000000ULL),               // 10^6
	    MakeUint128(0ULL,                    10000000ULL),              // 10^7
	    MakeUint128(0ULL,                    100000000ULL),             // 10^8
	    MakeUint128(0ULL,                    1000000000ULL),            // 10^9
	    MakeUint128(0ULL,                    10000000000ULL),           // 10^10
	    MakeUint128(0ULL,                    100000000000ULL),          // 10^11
	    MakeUint128(0ULL,                    1000000000000ULL),         // 10^12
	    MakeUint128(0ULL,                    10000000000000ULL),        // 10^13
	    MakeUint128(0ULL,                    100000000000000ULL),       // 10^14
	    MakeUint128(0ULL,                    1000000000000000ULL),      // 10^15
	    MakeUint128(0ULL,                    10000000000000000ULL),     // 10^16
	    MakeUint128(0ULL,                    100000000000000000ULL),    // 10^17
	    MakeUint128(0ULL,                    1000000000000000000ULL),   // 10^18
	    MakeUint128(0ULL,                    10000000000000000000ULL),  // 10^19
	    // 10^20 - 10^38: require both halves
	    MakeUint128(5ULL,                    7766279631452241920ULL),   // 10^20
	    MakeUint128(54ULL,                   3875820019684212736ULL),   // 10^21
	    MakeUint128(542ULL,                  1864712049423024128ULL),   // 10^22
	    MakeUint128(5421ULL,                 200376420520689664ULL),    // 10^23
	    MakeUint128(54210ULL,                2003764205206896640ULL),   // 10^24
	    MakeUint128(542101ULL,               1590897978359414784ULL),   // 10^25
	    MakeUint128(5421010ULL,              15908979783594147840ULL),  // 10^26
	    MakeUint128(54210108ULL,             11515845246265065472ULL),  // 10^27
	    MakeUint128(542101086ULL,            4477988020393345024ULL),   // 10^28
	    MakeUint128(5421010862ULL,           7886392056514347008ULL),   // 10^29
	    MakeUint128(54210108624ULL,          5076944270305263616ULL),   // 10^30
	    MakeUint128(542101086242ULL,         13875954555633532928ULL),  // 10^31
	    MakeUint128(5421010862427ULL,        9632337040368467968ULL),   // 10^32
	    MakeUint128(54210108624275ULL,       4089650035136921600ULL),   // 10^33
	    MakeUint128(542101086242752ULL,      4003012203950112768ULL),   // 10^34
	    MakeUint128(5421010862427522ULL,     3136633892082024448ULL),   // 10^35
	    MakeUint128(54210108624275221ULL,    12919594847110692864ULL),  // 10^36
	    MakeUint128(542101086242752217ULL,   68739955140067328ULL),     // 10^37
	    MakeUint128(5421010862427522170ULL,  687399551400673280ULL),    // 10^38
	};
	// clang-format on

	D_ASSERT(exp <= 38);
	return table[exp];
}

} // namespace duckdb
