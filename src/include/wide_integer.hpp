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
