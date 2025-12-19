package schema

// ThoughtContent represents thought/reasoning content from models that support
// extended thinking (e.g., Gemini 3 models). It contains an opaque signature that
// must be passed back in subsequent requests to maintain reasoning context.
// This is required for tool calls and multi-turn conversations with thinking models.
type ThoughtContent struct {
	// Text is the thought/reasoning text (may be empty if the model doesn't expose it).
	Text string
	// Signature is an opaque signature for the thought that must be passed back
	// in subsequent requests. This is required for Gemini 3+ models when using
	// tool calls or multi-turn conversations.
	Signature string
}

// AgentAction is the agent's action to take.
type AgentAction struct {
	Tool      string
	ToolInput string
	Log       string
	ToolID    string
	// ThoughtSignature is an opaque signature for Gemini 3+ models that must be
	// passed back exactly as received when sending conversation history.
	// For parallel function calls, only the first call will have a signature.
	ThoughtSignature string
	// ThoughtParts contains thought/reasoning parts from models that support
	// extended thinking (e.g., Gemini 3 models). These must be included in
	// subsequent messages to maintain reasoning context for tool calls.
	ThoughtParts []ThoughtContent
}

// AgentStep is a step of the agent.
type AgentStep struct {
	Action      AgentAction
	Observation string
}

// AgentFinish is the agent's return value.
type AgentFinish struct {
	ReturnValues map[string]any
	Log          string
}
