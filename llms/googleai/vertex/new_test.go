package vertex

import (
	"net/http"
	"testing"

	"github.com/tmc/langchaingo/llms/googleai"
	"google.golang.org/genai"
)

func TestNewWithOptions(t *testing.T) {
	// Since we can't control the genai.NewClient behavior in tests,
	// we'll skip these tests that require actual API interaction
	t.Skip("Skipping New() tests that require actual API credentials or mocked clients")
}

func TestDefaultOptions(t *testing.T) {
	opts := googleai.DefaultOptions()

	// Test default values
	if opts.DefaultModel != "gemini-2.0-flash" {
		t.Errorf("expected default model 'gemini-2.0-flash', got %q", opts.DefaultModel)
	}
	if opts.DefaultEmbeddingModel != "text-embedding-004" {
		t.Errorf("expected default embedding model 'text-embedding-004', got %q", opts.DefaultEmbeddingModel)
	}
	if opts.DefaultCandidateCount != 1 {
		t.Errorf("expected default candidate count 1, got %d", opts.DefaultCandidateCount)
	}
	if opts.DefaultMaxTokens != 2048 {
		t.Errorf("expected default max tokens 2048, got %d", opts.DefaultMaxTokens)
	}
	if opts.DefaultTemperature != 0.5 {
		t.Errorf("expected default temperature 0.5, got %f", opts.DefaultTemperature)
	}
	if opts.DefaultTopK != 3 {
		t.Errorf("expected default TopK 3, got %d", opts.DefaultTopK)
	}
	if opts.DefaultTopP != 0.95 {
		t.Errorf("expected default TopP 0.95, got %f", opts.DefaultTopP)
	}
	if opts.HarmThreshold != googleai.HarmBlockOnlyHigh {
		t.Errorf("expected default harm threshold HarmBlockOnlyHigh, got %v", opts.HarmThreshold)
	}
	if opts.Backend != genai.BackendGeminiAPI {
		t.Errorf("expected default backend BackendGeminiAPI, got %v", opts.Backend)
	}
}

func TestOptionsApplication(t *testing.T) { //nolint:funlen // comprehensive test
	// Test that options modify the default correctly
	defaultOpts := googleai.DefaultOptions()

	// Apply options
	googleai.WithDefaultModel("custom-model")(&defaultOpts)
	if defaultOpts.DefaultModel != "custom-model" {
		t.Errorf("WithDefaultModel did not update model")
	}

	googleai.WithDefaultEmbeddingModel("custom-embedding")(&defaultOpts)
	if defaultOpts.DefaultEmbeddingModel != "custom-embedding" {
		t.Errorf("WithDefaultEmbeddingModel did not update embedding model")
	}

	googleai.WithDefaultCandidateCount(3)(&defaultOpts)
	if defaultOpts.DefaultCandidateCount != 3 {
		t.Errorf("WithDefaultCandidateCount did not update candidate count")
	}

	googleai.WithDefaultMaxTokens(4096)(&defaultOpts)
	if defaultOpts.DefaultMaxTokens != 4096 {
		t.Errorf("WithDefaultMaxTokens did not update max tokens")
	}

	googleai.WithDefaultTemperature(0.9)(&defaultOpts)
	if defaultOpts.DefaultTemperature != 0.9 {
		t.Errorf("WithDefaultTemperature did not update temperature")
	}

	googleai.WithDefaultTopK(5)(&defaultOpts)
	if defaultOpts.DefaultTopK != 5 {
		t.Errorf("WithDefaultTopK did not update TopK")
	}

	googleai.WithDefaultTopP(0.99)(&defaultOpts)
	if defaultOpts.DefaultTopP != 0.99 {
		t.Errorf("WithDefaultTopP did not update TopP")
	}

	googleai.WithHarmThreshold(googleai.HarmBlockNone)(&defaultOpts)
	if defaultOpts.HarmThreshold != googleai.HarmBlockNone {
		t.Errorf("WithHarmThreshold did not update harm threshold")
	}

	googleai.WithCloudProject("my-project")(&defaultOpts)
	if defaultOpts.CloudProject != "my-project" {
		t.Errorf("WithCloudProject did not update project")
	}

	googleai.WithCloudLocation("europe-west1")(&defaultOpts)
	if defaultOpts.CloudLocation != "europe-west1" {
		t.Errorf("WithCloudLocation did not update location")
	}

	// Test API key
	googleai.WithAPIKey("test-key")(&defaultOpts)
	if defaultOpts.APIKey != "test-key" {
		t.Error("WithAPIKey did not update API key")
	}

	// Test backend
	googleai.WithBackend(genai.BackendVertexAI)(&defaultOpts)
	if defaultOpts.Backend != genai.BackendVertexAI {
		t.Error("WithBackend did not update backend")
	}
}

func TestHarmThresholdValues(t *testing.T) {
	// Test that harm threshold constants have expected values
	tests := []struct {
		name      string
		threshold googleai.HarmBlockThreshold
		expected  int32
	}{
		{"HarmBlockUnspecified", googleai.HarmBlockUnspecified, 0},
		{"HarmBlockLowAndAbove", googleai.HarmBlockLowAndAbove, 1},
		{"HarmBlockMediumAndAbove", googleai.HarmBlockMediumAndAbove, 2},
		{"HarmBlockOnlyHigh", googleai.HarmBlockOnlyHigh, 3},
		{"HarmBlockNone", googleai.HarmBlockNone, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int32(tt.threshold) != tt.expected {
				t.Errorf("expected %s to be %d, got %d", tt.name, tt.expected, int32(tt.threshold))
			}
		})
	}
}

func TestVertexStructure(t *testing.T) {
	// Test that Vertex has the expected fields
	// This is mainly for documentation and to catch breaking changes
	v := &Vertex{}

	// Check that fields exist by assignment (will fail to compile if missing)
	v.CallbacksHandler = nil
	v.client = nil
	v.opts = googleai.Options{}
	v.model = ""
}

func TestWithHTTPClientOption(t *testing.T) {
	opts := googleai.DefaultOptions()

	// Create a custom HTTP client
	httpClient := &http.Client{}

	// Apply the HTTP client option
	googleai.WithHTTPClient(httpClient)(&opts)

	// Verify that the HTTP client was set
	if opts.HTTPClient != httpClient {
		t.Error("WithHTTPClient did not set the HTTP client")
	}
}

func TestOptionsEnsureAuthPresent(t *testing.T) {
	tests := []struct {
		name           string
		opts           googleai.Options
		envAPIKey      string
		expectAddition bool
	}{
		{
			name: "with existing API key",
			opts: googleai.Options{
				APIKey: "existing-key",
			},
			envAPIKey:      "env-key",
			expectAddition: false,
		},
		{
			name:           "without API key but with env key",
			opts:           googleai.Options{},
			envAPIKey:      "env-key",
			expectAddition: true,
		},
		{
			name:           "without API key and no env key",
			opts:           googleai.Options{},
			envAPIKey:      "",
			expectAddition: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clear env vars first
			t.Setenv("GOOGLE_API_KEY", "")
			t.Setenv("GEMINI_API_KEY", "")

			// Set env var for test if needed
			if tt.envAPIKey != "" {
				t.Setenv("GOOGLE_API_KEY", tt.envAPIKey)
			}

			// Make a copy of opts to avoid modifying the original
			opts := tt.opts
			initialKey := opts.APIKey
			opts.EnsureAuthPresent()

			if tt.expectAddition {
				if opts.APIKey == initialKey || opts.APIKey == "" {
					t.Error("expected API key to be added from environment")
				}
			} else {
				if opts.APIKey != initialKey {
					t.Error("expected API key to not be changed")
				}
			}
		})
	}
}
