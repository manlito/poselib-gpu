// Not part of the public API: what tests and the benchmark need from a context.
#pragma once

#include "backend/backend.h"
#include "core/estimator.h"
#include "jpo/jpo.h"

namespace jpo {

// Hypotheses and rounds of the last call.
const BatchStats& ctx_stats(const jpo_ctx* ctx);
// The route of the context (to try other `auto` rules in the benchmark).
Route& ctx_route(jpo_ctx* ctx);
Backend* ctx_cpu(jpo_ctx* ctx);
Backend* ctx_gpu(jpo_ctx* ctx);

// The measured rule of a GPU backend on this kind of machine (core/auto_rule.cpp).
void auto_rule(int32_t gpu_backend, int cpu_threads, Route* route);

}  // namespace jpo
