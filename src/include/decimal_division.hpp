#pragma once

#include "wide_integer.hpp"

namespace duckdb {

// Perform decimal division with ROUND_HALF_UP rounding (Spark semantics).
//
// Given two scaled integers a and b (representing DECIMAL values),
// compute: result = (a * 10^scale_adj) / b, rounded HALF_UP.
//
// Returns the result as a signed __int128.
// Caller must handle division by zero before calling this function.
inline __int128 SparkDecimalDivide(__int128 a, __int128 b, uint32_t scale_adj) {
	// Handle signs separately, work with absolute values
	bool negative = (a < 0) != (b < 0);
	unsigned __int128 abs_a = Abs128(a);
	unsigned __int128 abs_b = Abs128(b);

	unsigned __int128 quotient;
	unsigned __int128 remainder;

	if (scale_adj == 0) {
		// No scaling needed
		quotient = abs_a / abs_b;
		remainder = abs_a % abs_b;
	} else {
		unsigned __int128 pow10_val = Pow10_128(scale_adj);

		// Check if abs_a * pow10_val would overflow unsigned __int128
		bool overflow = (abs_a != 0) &&
		                (pow10_val > (static_cast<unsigned __int128>(0) - 1) / abs_a);

		if (!overflow) {
			// Fast path: fits in 128 bits
			unsigned __int128 scaled = abs_a * pow10_val;
			quotient = scaled / abs_b;
			remainder = scaled % abs_b;
		} else {
			// Slow path: use 256-bit intermediate
			uint256_t scaled = Mul128(abs_a, pow10_val);
			quotient = Div256By128(scaled, abs_b, &remainder);
		}
	}

	// ROUND_HALF_UP: round away from zero when remainder >= half of divisor.
	// Equivalent to: if (2 * remainder >= abs_b) then round up.
	// Note: 2 * remainder cannot overflow unsigned __int128 because
	// remainder < abs_b <= 10^38, and 2 * 10^38 < 2^128.
	unsigned __int128 double_rem = remainder * 2;
	if (double_rem >= abs_b) {
		quotient++;
	}

	// Apply sign
	__int128 signed_result = static_cast<__int128>(quotient);
	return negative ? -signed_result : signed_result;
}

} // namespace duckdb
