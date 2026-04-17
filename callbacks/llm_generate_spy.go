package callbacks

import (
	"context"
	"sync"

	"github.com/tmc/langchaingo/llms"
)

// LLMGenerateSpy is a [Handler] that counts [Handler.HandleLLMGenerateContentStart] and
// [Handler.HandleLLMGenerateContentEnd] invocations. Other handler methods no-op via embedded
// [SimpleHandler]. It is intended for tests that assert LLM callback wiring (context vs struct field).
type LLMGenerateSpy struct {
	SimpleHandler

	mu     sync.Mutex
	starts int
	ends   int
}

var _ Handler = (*LLMGenerateSpy)(nil)

func (s *LLMGenerateSpy) HandleLLMGenerateContentStart(context.Context, []llms.MessageContent) {
	s.mu.Lock()
	s.starts++
	s.mu.Unlock()
}

func (s *LLMGenerateSpy) HandleLLMGenerateContentEnd(context.Context, *llms.ContentResponse) {
	s.mu.Lock()
	s.ends++
	s.mu.Unlock()
}

// Counts returns the number of LLM generate start and end callbacks observed.
func (s *LLMGenerateSpy) Counts() (starts, ends int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.starts, s.ends
}
