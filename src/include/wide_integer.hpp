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
// Uses binary long division (256 iterations).
inline unsigned __int128 Div256By128(uint256_t num, unsigned __int128 den,
                                     unsigned __int128 *remainder) {
	unsigned __int128 quot = 0;
	unsigned __int128 rem = 0;

	// Process all 256 bits from MSB to LSB
	for (int32_t bit_idx = 255; bit_idx >= 0; bit_idx--) {
		// Shift remainder left by 1
		rem <<= 1;

		// Bring in the next bit of the numerator
		unsigned __int128 word = (bit_idx >= 128) ? num.hi : num.lo;
		int32_t bit_pos = bit_idx % 128;
		if (word & (static_cast<unsigned __int128>(1) << bit_pos)) {
			rem |= 1;
		}

		// If remainder >= divisor, subtract and set quotient bit
		if (rem >= den) {
			rem -= den;
			if (bit_idx < 128) {
				quot |= (static_cast<unsigned __int128>(1) << bit_idx);
			}
			// If bit_idx >= 128, the quotient bit would overflow 128 bits.
			// For our use case (result precision <= 38), this should not happen.
			D_ASSERT(bit_idx < 128);
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

inline unsigned __int128 Pow10_128(uint32_t exp) {
	// For small exponents, fast lookup from a table
	static const uint64_t small_table[] = {
	    1ULL,
	    10ULL,
	    100ULL,
	    1000ULL,
	    10000ULL,
	    100000ULL,
	    1000000ULL,
	    10000000ULL,
	    100000000ULL,
	    1000000000ULL,
	    10000000000ULL,
	    100000000000ULL,
	    1000000000000ULL,
	    10000000000000ULL,
	    100000000000000ULL,
	    1000000000000000ULL,
	    10000000000000000ULL,
	    100000000000000000ULL,
	    1000000000000000000ULL,
	    10000000000000000000ULL, // 10^19
	};

	if (exp <= 19) {
		return static_cast<unsigned __int128>(small_table[exp]);
	}

	// For 10^20 through 10^38, compute from 10^19
	unsigned __int128 result = static_cast<unsigned __int128>(small_table[19]);
	for (uint32_t i = 19; i < exp; i++) {
		result *= 10;
	}
	return result;
}

} // namespace duckdb
