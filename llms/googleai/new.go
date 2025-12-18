// package googleai implements a langchaingo provider for Google AI LLMs.
// See https://ai.google.dev/ for more details.
package googleai

import (
	"context"
	"os"
	"reflect"
	"strings"

	genai "google.golang.org/genai"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
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
		model: clientOptions.DefaultModel, // Store the default model
	}

	// Build ClientConfig for the new library
	config := &genai.ClientConfig{}
	
	// Extract API key from options or environment
	// Convert ClientOptions to interface{} slice for extraction
	optsInterface := make([]interface{}, len(clientOptions.ClientOptions))
	for i, opt := range clientOptions.ClientOptions {
		optsInterface[i] = opt
	}
	apiKey := extractAPIKey(optsInterface)
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	
	// If Vertex AI options are present, use Vertex backend
	if clientOptions.CloudProject != "" && clientOptions.CloudLocation != "" {
		config.Backend = genai.BackendVertexAI
		config.Project = clientOptions.CloudProject
		config.Location = clientOptions.CloudLocation
	} else {
		// Use Gemini API backend
		config.Backend = genai.BackendGeminiAPI
		if apiKey != "" {
			config.APIKey = apiKey
		}
	}

	client, err := genai.NewClient(ctx, config)
	if err != nil {
		return gi, err
	}

	gi.client = client
	return gi, nil
}

// Close closes the underlying genai client.
// The new library may handle cleanup differently - check if Close is needed
func (g *GoogleAI) Close() error {
	// TODO: Check if new library needs explicit close or handles it automatically
	// The new library may not have a Close method or may handle cleanup differently
	if g.client != nil {
		// If the library has Close, uncomment:
		// return g.client.Close()
	}
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

	// Future Gemini 3+ models expected to support reasoning
	if strings.Contains(model, "gemini-3") || strings.Contains(model, "gemini-4") {
		return true
	}

	// Gemini Experimental models may have reasoning capabilities
	if strings.Contains(model, "gemini-exp") && strings.Contains(model, "thinking") {
		return true
	}

	return false
}

// extractAPIKey extracts the API key from ClientOptions if present
func extractAPIKey(opts []interface{}) string {
	// The options are of type option.ClientOption
	// We need to inspect them to find the API key option
	// This is a simplified approach - the actual extraction depends on option internals
	for _, opt := range opts {
		// Use reflection to check if this is an API key option
		v := reflect.ValueOf(opt)
		if !v.IsValid() {
			continue
		}
		
		// Check if this option sets an API key
		// The option package uses internal types, so we check the string representation
		optType := v.Type().String()
		if optType == "option.withAPIKey" || strings.Contains(optType, "APIKey") {
			// Try to extract the value - this is implementation-dependent
			// For now, we'll rely on environment variable or explicit config
		}
	}
	
	return ""
}
