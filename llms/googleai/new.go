// package googleai implements a langchaingo provider for Google AI LLMs.
// See https://ai.google.dev/ for more details.
package googleai

import (
	"context"
	"strings"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
	"google.golang.org/genai"
)

// GoogleAI is a type that represents a Google AI API client.
type GoogleAI struct {
	CallbacksHandler callbacks.Handler
	client           *genai.Client
	opts             Options
	model            string // Track current model for reasoning detection
}

var (
	_ llms.Model          = &GoogleAI{}
	_ llms.ReasoningModel = &GoogleAI{}
)

// New creates a new GoogleAI client.
func New(ctx context.Context, opts ...Option) (*GoogleAI, error) {
	clientOptions := DefaultOptions()
	for _, opt := range opts {
		opt(&clientOptions)
	}
	clientOptions.EnsureAuthPresent()

	gi := &GoogleAI{
		opts:  clientOptions,
		model: clientOptions.DefaultModel,
	}

	// Build the ClientConfig for the new SDK
	cc := &genai.ClientConfig{
		APIKey:     clientOptions.APIKey,
		Backend:    clientOptions.Backend,
		Project:    clientOptions.CloudProject,
		Location:   clientOptions.CloudLocation,
		HTTPClient: clientOptions.HTTPClient,
	}

	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return gi, err
	}

	gi.client = client
	return gi, nil
}

// Close closes the underlying genai client.
// This should be called when the GoogleAI instance is no longer needed
// to prevent memory leaks from the underlying connections.
func (g *GoogleAI) Close() error {
	// The new SDK client doesn't have a Close method
	// as it uses HTTP client pooling
	return nil
}

// SupportsReasoning implements the ReasoningModel interface.
// Returns true if the current model supports reasoning/thinking tokens.
func (g *GoogleAI) SupportsReasoning() bool {
	// Check the current model (may have been overridden by WithModel option)
	model := g.model
	if model == "" {
		model = g.opts.DefaultModel
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
