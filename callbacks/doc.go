// Package callbacks includes a standard interface for hooking into various
// stages of your LLM application. The package contains an implementation of
// this interface that prints to the standard output.
//
// Context-based handlers: use [ContextWithHandler] to attach [Handler] values to a
// [context.Context]. For [TimingHandler] span pairing, also use [EnsureTraceFrames]
//  or [ContextWithTraceFrames] so stack state is scoped per logical trace. Library 
// choke points (e.g. [github.com/tmc/langchaingo/chains.Call],
// agent executor, retriever) merge context-derived handlers with component-level
// handlers via [EffectiveHandler] (context first, then field handlers). Direct
// [github.com/tmc/langchaingo/tools.Tool] calls do not participate unless you route
// through an agent or merge handlers in application code.
package callbacks
