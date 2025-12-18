// package vertex implements a langchaingo provider for Google Vertex AI LLMs,
// including the Gemini models.
// See https://cloud.google.com/vertex-ai for more details.
package vertex

import (
	"context"
	"strings"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/googleai"
	"google.golang.org/genai"
)

// Vertex is a type that represents a Vertex AI API client.
type Vertex struct {
	CallbacksHandler callbacks.Handler
	client           *genai.Client
	opts             googleai.Options
	model            string // Track current model for reasoning detection
}

var (
	_ llms.Model          = &Vertex{}
	_ llms.ReasoningModel = &Vertex{}
)

// New creates a new Vertex client.
func New(ctx context.Context, opts ...googleai.Option) (*Vertex, error) {
	clientOptions := googleai.DefaultOptions()

	// Set Vertex AI as the default backend
	clientOptions.Backend = genai.BackendVertexAI

	for _, opt := range opts {
		opt(&clientOptions)
	}

	// Build the ClientConfig for the new unified SDK
	cc := &genai.ClientConfig{
		Backend:    genai.BackendVertexAI,
		Project:    clientOptions.CloudProject,
		Location:   clientOptions.CloudLocation,
		HTTPClient: clientOptions.HTTPClient,
	}

	// Use API key if provided (for Vertex AI Express mode)
	if clientOptions.APIKey != "" {
		cc.APIKey = clientOptions.APIKey
	}

	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return nil, err
	}

	v := &Vertex{
		opts:   clientOptions,
		client: client,
		model:  clientOptions.DefaultModel,
	}
	return v, nil
}

// Close closes the underlying genai client.
// This should be called when the Vertex instance is no longer needed
// to prevent memory leaks from the underlying connections.
func (v *Vertex) Close() error {
	// The new SDK client doesn't have a Close method
	// as it uses HTTP client pooling
	return nil
}

// SupportsReasoning implements the ReasoningModel interface.
// Returns true if the current model supports reasoning/thinking tokens.
func (v *Vertex) SupportsReasoning() bool {
	// Check the current model (may have been overridden by WithModel option)
	model := v.model
	if model == "" {
		model = v.opts.DefaultModel
	}

	// Gemini 2.0 models support reasoning/thinking capabilities
	if strings.Contains(model, "gemini-2.0") {
		return true
	}

	// Gemini 2.5 and 3+ models support reasoning
	if strings.Contains(model, "gemini-2.5") || strings.Contains(model, "gemini-3") || strings.Contains(model, "gemini-4") {
		return true
	}

	// Gemini Experimental models may have reasoning capabilities
	if strings.Contains(model, "gemini-exp") && strings.Contains(model, "thinking") {
		return true
	}

	return false
}
