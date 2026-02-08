#pragma once

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/aggregate_state.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "spark_precision.hpp"
#include "wide_integer.hpp"
#include "decimal_division.hpp"

namespace duckdb {

// ============================================================================
// Bind data for spark_sum and spark_avg (stores the input scale for finalize)
// ============================================================================

struct SparkAggBindData : public FunctionData {
	uint8_t input_scale;
	uint8_t result_scale;

	SparkAggBindData(uint8_t input_scale_p, uint8_t result_scale_p)
	    : input_scale(input_scale_p), result_scale(result_scale_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<SparkAggBindData>(input_scale, result_scale);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<SparkAggBindData>();
		return input_scale == other.input_scale && result_scale == other.result_scale;
	}
};

// ============================================================================
// Helper: Convert __int128 result to the target DECIMAL physical type
// ============================================================================

template <typename T>
static inline void WriteAggResult(T &target, __int128 val) {
	target = static_cast<T>(val);
}

template <>
inline void WriteAggResult<hugeint_t>(hugeint_t &target, __int128 val) {
	target = Int128ToHugeint(val);
}

// ============================================================================
// spark_sum: DECIMAL path
//
// Accumulates values into hugeint_t (scaled integers).
// Input is promoted to DECIMAL(38, s) by DuckDB's implicit cast.
// Returns DECIMAL(min(p+10, 38), s) per Spark rules.
// ============================================================================

struct SparkSumDecimalState {
	hugeint_t value;
	bool isset;

	void Initialize() {
		isset = false;
		value = hugeint_t(0);
	}

	void Combine(const SparkSumDecimalState &other) {
		if (other.isset) {
			isset = true;
			value += other.value;
		}
	}
};

// Templatized operation so Finalize can target different physical types
template <typename RESULT_TYPE>
struct SparkSumDecimalOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.Initialize();
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &) {
		state.isset = true;
		state.value += hugeint_t(input);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &, idx_t count) {
		state.isset = true;
		state.value += hugeint_t(input) * Hugeint::Convert(static_cast<int64_t>(count));
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.Combine(source);
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.isset) {
			finalize_data.ReturnNull();
		} else {
			__int128 val = HugeintToInt128(state.value);
			WriteAggResult(target, val);
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

// Helper: create a SparkSumDecimal AggregateFunction for a specific result physical type
template <typename RESULT_TYPE>
static AggregateFunction GetSparkSumDecimalFunction() {
	return AggregateFunction::UnaryAggregate<SparkSumDecimalState, hugeint_t, RESULT_TYPE,
	                                          SparkSumDecimalOperation<RESULT_TYPE>>(
	    LogicalType::DECIMAL(38, 0), LogicalType::DECIMAL(38, 0));
}

static unique_ptr<FunctionData> BindSparkSumDecimal(ClientContext &context, AggregateFunction &function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	auto &type = arguments[0]->return_type;
	if (type.id() != LogicalTypeId::DECIMAL) {
		throw InvalidInputException("spark_sum DECIMAL overload requires DECIMAL argument");
	}

	uint8_t p = DecimalType::GetWidth(type);
	uint8_t s = DecimalType::GetScale(type);
	auto result = ComputeSumType(p, s);

	// Promote input to DECIMAL(38, s) -> hugeint_t physical type
	function.arguments[0] = LogicalType::DECIMAL(38, s);
	auto result_type = LogicalType::DECIMAL(result.precision, result.scale);
	function.return_type = result_type;

	// Select the correct function implementation based on result physical type
	// The finalize function must write to the correct physical type for the result DECIMAL.
	{
		AggregateFunction tf = [&]() -> AggregateFunction {
			switch (result_type.InternalType()) {
			case PhysicalType::INT16:
				return GetSparkSumDecimalFunction<int16_t>();
			case PhysicalType::INT32:
				return GetSparkSumDecimalFunction<int32_t>();
			case PhysicalType::INT64:
				return GetSparkSumDecimalFunction<int64_t>();
			case PhysicalType::INT128:
				return GetSparkSumDecimalFunction<hugeint_t>();
			default:
				throw InternalException("Unexpected physical type for spark_sum DECIMAL result");
			}
		}();
		function.update = tf.update;
		function.combine = tf.combine;
		function.finalize = tf.finalize;
		function.simple_update = tf.simple_update;
	}

	return make_uniq<SparkAggBindData>(s, result.scale);
}

// ============================================================================
// spark_sum: Integer path
//
// Spark: SUM(int/long/short/byte) -> BIGINT
// Accumulates into int64_t, returns BIGINT.
// ============================================================================

struct SparkSumIntegerState {
	int64_t value;
	bool isset;

	void Initialize() {
		isset = false;
		value = 0;
	}

	void Combine(const SparkSumIntegerState &other) {
		if (other.isset) {
			isset = true;
			value += other.value;
		}
	}
};

struct SparkSumIntegerOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.Initialize();
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &) {
		state.isset = true;
		state.value += static_cast<int64_t>(input);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &, idx_t count) {
		state.isset = true;
		state.value += static_cast<int64_t>(input) * static_cast<int64_t>(count);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.Combine(source);
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.isset) {
			finalize_data.ReturnNull();
		} else {
			target = state.value;
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

// ============================================================================
// spark_avg: DECIMAL path
//
// Accumulates sum (hugeint_t) and count (uint64_t).
// At finalize, divides sum/count using SparkDecimalDivide with ROUND_HALF_UP.
// Returns DECIMAL(min(p+4, 38), min(s+4, 18)) per Spark rules.
// ============================================================================

struct SparkAvgDecimalState {
	hugeint_t sum;
	uint64_t count;

	void Initialize() {
		count = 0;
		sum = hugeint_t(0);
	}

	void Combine(const SparkAvgDecimalState &other) {
		count += other.count;
		sum += other.sum;
	}
};

// Templatized so Finalize can target different physical result types
template <typename RESULT_TYPE>
struct SparkAvgDecimalOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.Initialize();
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &) {
		state.count++;
		state.sum += hugeint_t(input);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &, idx_t count) {
		state.count += count;
		state.sum += hugeint_t(input) * Hugeint::Convert(static_cast<int64_t>(count));
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.Combine(source);
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (state.count == 0) {
			finalize_data.ReturnNull();
			return;
		}

		// Get bind data for scale adjustment
		auto &bind_data = finalize_data.input.bind_data->Cast<SparkAggBindData>();

		// scale_adj = result_scale - input_scale
		uint32_t scale_adj = static_cast<uint32_t>(bind_data.result_scale) -
		                     static_cast<uint32_t>(bind_data.input_scale);

		__int128 sum_val = HugeintToInt128(state.sum);
		__int128 count_val = static_cast<__int128>(state.count);

		unsigned __int128 pow10_val = (scale_adj > 0) ? Pow10_128(scale_adj) : 0;
		__int128 result = SparkDecimalDivide(sum_val, count_val, pow10_val);

		WriteAggResult(target, result);
	}

	static bool IgnoreNull() {
		return true;
	}
};

// Helper: create a SparkAvgDecimal AggregateFunction for a specific result physical type
template <typename RESULT_TYPE>
static AggregateFunction GetSparkAvgDecimalFunction() {
	return AggregateFunction::UnaryAggregate<SparkAvgDecimalState, hugeint_t, RESULT_TYPE,
	                                          SparkAvgDecimalOperation<RESULT_TYPE>>(
	    LogicalType::DECIMAL(38, 0), LogicalType::DECIMAL(38, 0));
}

static unique_ptr<FunctionData> BindSparkAvgDecimal(ClientContext &context, AggregateFunction &function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	auto &type = arguments[0]->return_type;
	if (type.id() != LogicalTypeId::DECIMAL) {
		throw InvalidInputException("spark_avg DECIMAL overload requires DECIMAL argument");
	}

	uint8_t p = DecimalType::GetWidth(type);
	uint8_t s = DecimalType::GetScale(type);
	auto result = ComputeAvgType(p, s);

	// Promote input to DECIMAL(38, s) -> hugeint_t physical type
	function.arguments[0] = LogicalType::DECIMAL(38, s);
	auto result_type = LogicalType::DECIMAL(result.precision, result.scale);
	function.return_type = result_type;

	// Select the correct function implementation based on result physical type
	{
		AggregateFunction tf = [&]() -> AggregateFunction {
			switch (result_type.InternalType()) {
			case PhysicalType::INT16:
				return GetSparkAvgDecimalFunction<int16_t>();
			case PhysicalType::INT32:
				return GetSparkAvgDecimalFunction<int32_t>();
			case PhysicalType::INT64:
				return GetSparkAvgDecimalFunction<int64_t>();
			case PhysicalType::INT128:
				return GetSparkAvgDecimalFunction<hugeint_t>();
			default:
				throw InternalException("Unexpected physical type for spark_avg DECIMAL result");
			}
		}();
		function.update = tf.update;
		function.combine = tf.combine;
		function.finalize = tf.finalize;
		function.simple_update = tf.simple_update;
	}

	return make_uniq<SparkAggBindData>(s, result.scale);
}

// spark_count is NOT needed as a separate extension function.
// DuckDB's built-in COUNT already returns BIGINT, matching Spark semantics.

// ============================================================================
// Factory functions to create the AggregateFunctionSets
// ============================================================================

inline AggregateFunctionSet CreateSparkSumFunctionSet() {
	AggregateFunctionSet set("spark_sum");

	// DECIMAL overload: input DECIMAL -> result DECIMAL(min(p+10,38), s)
	// Initial template uses hugeint_t; bind function swaps to correct physical type
	auto decimal_func = GetSparkSumDecimalFunction<hugeint_t>();
	decimal_func.bind = BindSparkSumDecimal;
	decimal_func.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	set.AddFunction(decimal_func);

	// Integer overloads: all return BIGINT (Spark semantics)
	// TINYINT
	auto tinyint_func = AggregateFunction::UnaryAggregate<SparkSumIntegerState, int8_t, int64_t,
	                                                       SparkSumIntegerOperation>(
	    LogicalType::TINYINT, LogicalType::BIGINT);
	tinyint_func.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	set.AddFunction(tinyint_func);

	// SMALLINT
	auto smallint_func = AggregateFunction::UnaryAggregate<SparkSumIntegerState, int16_t, int64_t,
	                                                        SparkSumIntegerOperation>(
	    LogicalType::SMALLINT, LogicalType::BIGINT);
	smallint_func.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	set.AddFunction(smallint_func);

	// INTEGER
	auto int_func = AggregateFunction::UnaryAggregate<SparkSumIntegerState, int32_t, int64_t,
	                                                   SparkSumIntegerOperation>(
	    LogicalType::INTEGER, LogicalType::BIGINT);
	int_func.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	set.AddFunction(int_func);

	// BIGINT
	auto bigint_func = AggregateFunction::UnaryAggregate<SparkSumIntegerState, int64_t, int64_t,
	                                                      SparkSumIntegerOperation>(
	    LogicalType::BIGINT, LogicalType::BIGINT);
	bigint_func.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	set.AddFunction(bigint_func);

	return set;
}

inline AggregateFunctionSet CreateSparkAvgFunctionSet() {
	AggregateFunctionSet set("spark_avg");

	// DECIMAL overload: input DECIMAL -> result DECIMAL(min(p+4,38), min(s+4,18))
	// Initial template uses hugeint_t; bind function swaps to correct physical type
	auto decimal_func = GetSparkAvgDecimalFunction<hugeint_t>();
	decimal_func.bind = BindSparkAvgDecimal;
	decimal_func.order_dependent = AggregateOrderDependent::NOT_ORDER_DEPENDENT;
	set.AddFunction(decimal_func);

	return set;
}

// No CreateSparkCountFunctionSet — DuckDB COUNT already matches Spark.

} // namespace duckdb
