#define DUCKDB_EXTENSION_MAIN

#include "spark_decimal_div_extension.hpp"
#include "spark_precision.hpp"
#include "decimal_division.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

// ---------------------------------------------------------------------------
// Write helpers: convert __int128 to target physical type
// ---------------------------------------------------------------------------

template <typename T>
static inline void WriteResult(T *data, idx_t idx, __int128 val) {
	data[idx] = static_cast<T>(val);
}

// Specialization for hugeint_t
template <>
inline void WriteResult<hugeint_t>(hugeint_t *data, idx_t idx, __int128 val) {
	data[idx] = Int128ToHugeint(val);
}

// ---------------------------------------------------------------------------
// Execution function template
// ---------------------------------------------------------------------------
// RESULT_TYPE is the physical C++ type for the output DECIMAL
// (int16_t, int32_t, int64_t, or hugeint_t).
// Inputs are always hugeint_t because we promote both arguments to DECIMAL(38, s).

template <typename RESULT_TYPE>
static void SparkDivExec(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = func_expr.bind_info->Cast<SparkDivBindData>();
	uint32_t scale_adj = bind_data.scale_adj;

	idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<RESULT_TYPE>(result);
	auto &result_validity = FlatVector::Validity(result);

	UnifiedVectorFormat a_fmt, b_fmt;
	args.data[0].ToUnifiedFormat(count, a_fmt);
	args.data[1].ToUnifiedFormat(count, b_fmt);

	auto a_data = UnifiedVectorFormat::GetData<hugeint_t>(a_fmt);
	auto b_data = UnifiedVectorFormat::GetData<hugeint_t>(b_fmt);

	for (idx_t i = 0; i < count; i++) {
		auto a_idx = a_fmt.sel->get_index(i);
		auto b_idx = b_fmt.sel->get_index(i);

		// NULL propagation
		if (!a_fmt.validity.RowIsValid(a_idx) || !b_fmt.validity.RowIsValid(b_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		__int128 b_val = HugeintToInt128(b_data[b_idx]);

		// Division by zero -> NULL
		if (b_val == 0) {
			result_validity.SetInvalid(i);
			continue;
		}

		__int128 a_val = HugeintToInt128(a_data[a_idx]);
		__int128 div_result = SparkDecimalDivide(a_val, b_val, scale_adj);

		// Write result, converting from __int128 to the target physical type
		WriteResult(result_data, i, div_result);
	}
}

// ---------------------------------------------------------------------------
// Bind function: resolve types and select implementation
// ---------------------------------------------------------------------------

static unique_ptr<FunctionData> BindSparkDecimalDiv(ClientContext &context,
                                                     ScalarFunction &bound_function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	auto &type_a = arguments[0]->return_type;
	auto &type_b = arguments[1]->return_type;

	if (type_a.id() != LogicalTypeId::DECIMAL || type_b.id() != LogicalTypeId::DECIMAL) {
		throw InvalidInputException("spark_decimal_div requires DECIMAL arguments");
	}

	uint8_t p1 = DecimalType::GetWidth(type_a);
	uint8_t s1 = DecimalType::GetScale(type_a);
	uint8_t p2 = DecimalType::GetWidth(type_b);
	uint8_t s2 = DecimalType::GetScale(type_b);

	// Compute result type per Spark 4.1 rules
	auto result = ComputeDivisionType(p1, s1, p2, s2);

	// scale_adj = result_scale - s1 + s2
	// This is always >= 0 for valid Spark inputs
	uint32_t scale_adj = static_cast<uint32_t>(result.scale) - static_cast<uint32_t>(s1) + static_cast<uint32_t>(s2);

	// Promote both inputs to DECIMAL(38, s_original) -> hugeint_t physical type
	// DuckDB will insert implicit casts
	bound_function.arguments[0] = LogicalType::DECIMAL(38, s1);
	bound_function.arguments[1] = LogicalType::DECIMAL(38, s2);

	// Set result type
	auto result_type = LogicalType::DECIMAL(result.precision, result.scale);
	bound_function.return_type = result_type;

	// Select implementation based on result physical type
	switch (result_type.InternalType()) {
	case PhysicalType::INT16:
		bound_function.function = SparkDivExec<int16_t>;
		break;
	case PhysicalType::INT32:
		bound_function.function = SparkDivExec<int32_t>;
		break;
	case PhysicalType::INT64:
		bound_function.function = SparkDivExec<int64_t>;
		break;
	case PhysicalType::INT128:
		bound_function.function = SparkDivExec<hugeint_t>;
		break;
	default:
		throw InternalException("Unexpected physical type for DECIMAL result");
	}

	return make_uniq<SparkDivBindData>(scale_adj);
}

// ---------------------------------------------------------------------------
// Internal loading logic
// ---------------------------------------------------------------------------

static void LoadInternal(ExtensionLoader &loader) {
	vector<LogicalType> args = {LogicalType::ANY, LogicalType::ANY};
	ScalarFunction func("spark_decimal_div", std::move(args), LogicalType::ANY, SparkDivExec<hugeint_t>,
	                    BindSparkDecimalDiv);
	func.null_handling = FunctionNullHandling::SPECIAL_HANDLING;

	loader.RegisterFunction(func);
}

// ---------------------------------------------------------------------------
// Extension class methods
// ---------------------------------------------------------------------------

void SparkDecimalDivExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string SparkDecimalDivExtension::Name() {
	return "spark_decimal_div";
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(spark_decimal_div, loader) {
	duckdb::LoadInternal(loader);
}

DUCKDB_EXTENSION_API const char *spark_decimal_div_version() {
	return duckdb::DuckDB::LibraryVersion();
}

}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
