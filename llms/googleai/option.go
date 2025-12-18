package googleai

import (
	"net/http"
	"os"

	"github.com/tmc/langchaingo/llms"
	"google.golang.org/genai"
)

// Options is a set of options for GoogleAI and Vertex clients.
type Options struct {
	// APIKey is the API key for the Gemini API.
	// Can also be set via the GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
	APIKey string

	// CloudProject is the GCP project ID for Vertex AI.
	// Can also be set via the GOOGLE_CLOUD_PROJECT environment variable.
	CloudProject string

	// CloudLocation is the GCP location/region for Vertex AI.
	// Can also be set via the GOOGLE_CLOUD_LOCATION or GOOGLE_CLOUD_REGION environment variable.
	CloudLocation string

	// Backend specifies which backend to use (Gemini API or Vertex AI).
	// Defaults to BackendGeminiAPI.
	Backend genai.Backend

	// DefaultModel is the default model to use for content generation.
	DefaultModel string

	// DefaultEmbeddingModel is the default model to use for embeddings.
	DefaultEmbeddingModel string

	// DefaultCandidateCount is the default number of candidates to generate.
	DefaultCandidateCount int

	// DefaultMaxTokens is the default maximum number of tokens to generate.
	DefaultMaxTokens int

	// DefaultTemperature is the default temperature for generation.
	DefaultTemperature float64

	// DefaultTopK is the default top-k value for generation.
	DefaultTopK int

	// DefaultTopP is the default top-p value for generation.
	DefaultTopP float64

	// HarmThreshold is the safety/harm setting for the model.
	HarmThreshold HarmBlockThreshold

	// HTTPClient is an optional custom HTTP client to use.
	HTTPClient *http.Client
}

// DefaultOptions returns the default options for GoogleAI.
func DefaultOptions() Options {
	return Options{
		Backend:               genai.BackendGeminiAPI,
		DefaultModel:          "gemini-2.0-flash",
		DefaultEmbeddingModel: "text-embedding-004",
		DefaultCandidateCount: 1,
		DefaultMaxTokens:      2048,
		DefaultTemperature:    0.5,
		DefaultTopK:           3,
		DefaultTopP:           0.95,
		HarmThreshold:         HarmBlockOnlyHigh,
	}
}

// EnsureAuthPresent attempts to ensure that the client has authentication information.
// If it does not, it will attempt to use the GOOGLE_API_KEY environment variable.
func (o *Options) EnsureAuthPresent() {
	if o.APIKey == "" {
		if key := os.Getenv("GOOGLE_API_KEY"); key != "" {
			o.APIKey = key
		} else if key := os.Getenv("GEMINI_API_KEY"); key != "" {
			o.APIKey = key
		}
	}
}

// Option is a function that configures Options.
type Option func(*Options)

// WithAPIKey passes the API KEY (token) to the client.
func WithAPIKey(apiKey string) Option {
	return func(opts *Options) {
		opts.APIKey = apiKey
	}
}

// WithHTTPClient sets a custom HTTP client to use for requests.
func WithHTTPClient(httpClient *http.Client) Option {
	return func(opts *Options) {
		opts.HTTPClient = httpClient
	}
}

// WithCloudProject passes the GCP cloud project name to the client.
// This is useful for Vertex AI clients.
func WithCloudProject(p string) Option {
	return func(opts *Options) {
		opts.CloudProject = p
	}
}

// WithCloudLocation passes the GCP cloud location (region) name to the client.
// This is useful for Vertex AI clients.
func WithCloudLocation(l string) Option {
	return func(opts *Options) {
		opts.CloudLocation = l
	}
}

// WithBackend sets the backend to use (Gemini API or Vertex AI).
func WithBackend(backend genai.Backend) Option {
	return func(opts *Options) {
		opts.Backend = backend
	}
}

// WithDefaultModel passes a default content model name to the client.
// This model name is used if not explicitly provided in specific client invocations.
func WithDefaultModel(defaultModel string) Option {
	return func(opts *Options) {
		opts.DefaultModel = defaultModel
	}
}

// WithDefaultEmbeddingModel passes a default embedding model name to the client.
// This model name is used if not explicitly provided in specific client invocations.
func WithDefaultEmbeddingModel(defaultEmbeddingModel string) Option {
	return func(opts *Options) {
		opts.DefaultEmbeddingModel = defaultEmbeddingModel
	}
}

// WithDefaultCandidateCount sets the candidate count for the model.
func WithDefaultCandidateCount(defaultCandidateCount int) Option {
	return func(opts *Options) {
		opts.DefaultCandidateCount = defaultCandidateCount
	}
}

// WithDefaultMaxTokens sets the maximum token count for the model.
func WithDefaultMaxTokens(maxTokens int) Option {
	return func(opts *Options) {
		opts.DefaultMaxTokens = maxTokens
	}
}

// WithDefaultTemperature sets the temperature for the model.
func WithDefaultTemperature(defaultTemperature float64) Option {
	return func(opts *Options) {
		opts.DefaultTemperature = defaultTemperature
	}
}

// WithDefaultTopK sets the TopK for the model.
func WithDefaultTopK(defaultTopK int) Option {
	return func(opts *Options) {
		opts.DefaultTopK = defaultTopK
	}
}

// WithDefaultTopP sets the TopP for the model.
func WithDefaultTopP(defaultTopP float64) Option {
	return func(opts *Options) {
		opts.DefaultTopP = defaultTopP
	}
}

// WithHarmThreshold sets the safety/harm setting for the model, potentially
// limiting any harmful content it may generate.
func WithHarmThreshold(ht HarmBlockThreshold) Option {
	return func(opts *Options) {
		opts.HarmThreshold = ht
	}
}

// WithCachedContent enables the use of pre-created cached content.
// The cached content must be created separately using the Caches API.
func WithCachedContent(name string) llms.CallOption {
	return func(o *llms.CallOptions) {
		if o.Metadata == nil {
			o.Metadata = make(map[string]interface{})
		}
		o.Metadata["CachedContentName"] = name
	}
}

// HarmBlockThreshold is the threshold for blocking harmful content.
type HarmBlockThreshold int32

const (
	// HarmBlockUnspecified means threshold is unspecified.
	HarmBlockUnspecified HarmBlockThreshold = 0
	// HarmBlockLowAndAbove means content with NEGLIGIBLE will be allowed.
	HarmBlockLowAndAbove HarmBlockThreshold = 1
	// HarmBlockMediumAndAbove means content with NEGLIGIBLE and LOW will be allowed.
	HarmBlockMediumAndAbove HarmBlockThreshold = 2
	// HarmBlockOnlyHigh means content with NEGLIGIBLE, LOW, and MEDIUM will be allowed.
	HarmBlockOnlyHigh HarmBlockThreshold = 3
	// HarmBlockNone means all content will be allowed.
	HarmBlockNone HarmBlockThreshold = 4
)

// toGenAIHarmBlockThreshold converts HarmBlockThreshold to genai.HarmBlockThreshold.
func (h HarmBlockThreshold) toGenAIHarmBlockThreshold() genai.HarmBlockThreshold {
	switch h {
	case HarmBlockLowAndAbove:
		return genai.HarmBlockThresholdBlockLowAndAbove
	case HarmBlockMediumAndAbove:
		return genai.HarmBlockThresholdBlockMediumAndAbove
	case HarmBlockOnlyHigh:
		return genai.HarmBlockThresholdBlockOnlyHigh
	case HarmBlockNone:
		return genai.HarmBlockThresholdBlockNone
	default:
		return genai.HarmBlockThresholdUnspecified
	}
}
