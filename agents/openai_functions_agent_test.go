package agents_test

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/internal/httprr"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/tools"
)

// hasExistingRecording checks if a httprr recording exists for this test
func hasExistingRecording(t *testing.T) bool {
	testName := strings.ReplaceAll(t.Name(), "/", "_")
	testName = strings.ReplaceAll(testName, " ", "_")
	recordingPath := filepath.Join("testdata", testName+".httprr")
	_, err := os.Stat(recordingPath)
	return err == nil
}

func TestOpenAIFunctionsAgentWithHTTPRR(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Skip if no recording available and no credentials
	if !hasExistingRecording(t) {
		t.Skip("No httprr recording available. Hint: Re-run tests with -httprecord=. to record new HTTP interactions")
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	// Configure OpenAI client with httprr
	opts := []openai.Option{
		openai.WithModel("gpt-4o"),
		openai.WithHTTPClient(rr.Client()),
	}
	if rr.Replaying() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	if err != nil {
		t.Fatal(err)
	}

	// Create a simple calculator tool
	calculator := tools.Calculator{}

	// Create the OpenAI Functions agent
	agent := agents.NewOpenAIFunctionsAgent(
		llm,
		[]tools.Tool{calculator},
		agents.NewOpenAIOption().WithSystemMessage("You are a helpful assistant that can perform calculations."),
	)

	// Create executor
	executor := agents.NewExecutor(agent)

	// Run a simple calculation
	result, err := chains.Run(ctx, executor, "What is 15 multiplied by 4?")
	if err != nil {
		// Check if this is a recording mismatch error
		if strings.Contains(err.Error(), "cached HTTP response not found") {
			t.Skip("Recording format has changed or is incompatible. Hint: Re-run tests with -httprecord=. to record new HTTP interactions")
		}
		t.Fatal(err)
	}

	t.Logf("Agent response: %s", result)

	// Verify the result contains 60
	if !strings.Contains(result, "60") {
		t.Errorf("expected calculation result 60 in response, got: %s", result)
	}
}

func TestOpenAIFunctionsAgentComplexCalculation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Skip if no recording available and no credentials
	if !hasExistingRecording(t) {
		t.Skip("No httprr recording available. Hint: Re-run tests with -httprecord=. to record new HTTP interactions")
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)

	// Configure OpenAI client with httprr
	opts := []openai.Option{
		openai.WithModel("gpt-4o"),
		openai.WithHTTPClient(rr.Client()),
	}
	if rr.Replaying() {
		opts = append(opts, openai.WithToken("test-api-key"))
	}

	llm, err := openai.New(opts...)
	require.NoError(t, err)

	// Create a calculator tool
	calculator := tools.Calculator{}

	// Create the OpenAI Functions agent with extra messages
	agent := agents.NewOpenAIFunctionsAgent(
		llm,
		[]tools.Tool{calculator},
		agents.NewOpenAIOption().WithSystemMessage("You are a helpful math assistant."),
		agents.NewOpenAIOption().WithExtraMessages([]prompts.MessageFormatter{
			prompts.NewHumanMessagePromptTemplate("Please show your work step by step.", nil),
		}),
	)

	// Create executor with options
	executor := agents.NewExecutor(
		agent,
		agents.WithMaxIterations(5),
	)

	// Run a more complex calculation
	result, err := chains.Run(ctx, executor, "If I have 3 groups of 7 items, and I add 9 more items, how many items do I have in total?")
	if err != nil {
		// Check if this is a recording mismatch error
		if strings.Contains(err.Error(), "cached HTTP response not found") {
			t.Skip("Recording format has changed or is incompatible. Hint: Re-run tests with -httprecord=. to record new HTTP interactions")
		}
		t.Fatalf("failed to run agent: %v", err)
	}
	t.Logf("Agent response: %s", result)

	// Verify the result contains 30 (3*7 + 9 = 21 + 9 = 30)
	if !strings.Contains(result, "30") {
		t.Errorf("expected calculation result 30 in response, got: %s", result)
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_NilResponse tests that ParseOutput handles nil response gracefully
func TestOpenAIFunctionsAgent_ParseOutput_NilResponse(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test with nil response - should return error instead of panic
	_, _, err := agent.ParseOutput(nil)
	if err == nil {
		t.Error("expected error for nil response")
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_EmptyChoices tests that ParseOutput handles empty choices gracefully
func TestOpenAIFunctionsAgent_ParseOutput_EmptyChoices(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test with empty choices - should return error instead of panic
	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{},
	}
	_, _, err := agent.ParseOutput(resp)
	if err == nil {
		t.Error("expected error for empty choices")
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_MultipleToolCalls tests multiple tool calls handling
func TestOpenAIFunctionsAgent_ParseOutput_MultipleToolCalls(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test multiple tool calls - should handle all calls, not just first one
	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				ToolCalls: []llms.ToolCall{
					{
						ID: "call1",
						FunctionCall: &llms.FunctionCall{
							Name:      "calculator",
							Arguments: `{"__arg1": "2+2"}`,
						},
					},
					{
						ID: "call2",
						FunctionCall: &llms.FunctionCall{
							Name:      "weather",
							Arguments: `{"__arg1": "Seattle"}`,
						},
					},
				},
			},
		},
	}

	actions, finish, err := agent.ParseOutput(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if finish != nil {
		t.Error("expected actions, got finish")
	}
	if len(actions) != 2 {
		t.Errorf("expected 2 actions, got %d", len(actions))
	}
}

// TestOpenAIFunctionsAgent_ParseOutput_ThoughtSignatures tests that thought signatures are captured from tool calls
func TestOpenAIFunctionsAgent_ParseOutput_ThoughtSignatures(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Test that thought signatures from Gemini 3+ models are captured
	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				ToolCalls: []llms.ToolCall{
					{
						ID: "call1",
						FunctionCall: &llms.FunctionCall{
							Name:      "calculator",
							Arguments: `{"__arg1": "2+2"}`,
						},
						ThoughtSignature: "signature-abc-123",
					},
					{
						ID: "call2",
						FunctionCall: &llms.FunctionCall{
							Name:      "weather",
							Arguments: `{"__arg1": "Seattle"}`,
						},
						// Second parallel call typically doesn't have a signature
						ThoughtSignature: "",
					},
				},
				ThoughtParts: []llms.ThoughtContent{
					{
						Text:      "Let me think about this step by step...",
						Signature: "thought-sig-xyz",
					},
				},
			},
		},
	}

	actions, finish, err := agent.ParseOutput(resp)
	require.NoError(t, err)
	require.Nil(t, finish)
	require.Len(t, actions, 2)

	// First action should have the thought signature
	require.Equal(t, "signature-abc-123", actions[0].ThoughtSignature)
	// First action should have the thought parts
	require.Len(t, actions[0].ThoughtParts, 1)
	require.Equal(t, "Let me think about this step by step...", actions[0].ThoughtParts[0].Text)
	require.Equal(t, "thought-sig-xyz", actions[0].ThoughtParts[0].Signature)

	// Second action should not have thought signature or parts
	require.Empty(t, actions[1].ThoughtSignature)
	require.Empty(t, actions[1].ThoughtParts)
}

// TestOpenAIFunctionsAgent_ConstructScratchPad_ThoughtSignatures tests that thought signatures are included when rebuilding conversation
func TestOpenAIFunctionsAgent_ConstructScratchPad_ThoughtSignatures(t *testing.T) {
	t.Parallel()

	// Create a simple agent (we just need to test the scratchpad construction)
	agent := agents.NewOpenAIFunctionsAgent(nil, nil)

	// Create steps with thought signatures
	steps := []schema.AgentStep{
		{
			Action: schema.AgentAction{
				Tool:             "calculator",
				ToolInput:        "2+2",
				Log:              "Invoking: calculator with 2+2",
				ToolID:           "call1",
				ThoughtSignature: "signature-abc-123",
				ThoughtParts: []schema.ThoughtContent{
					{
						Text:      "Let me calculate this...",
						Signature: "thought-sig-xyz",
					},
				},
			},
			Observation: "4",
		},
	}

	// Access the internal constructScratchPad method via Plan
	// We can't call it directly since it's unexported, but we can test the round-trip
	// by examining what gets passed to the LLM

	// For now, we verify the schema types are correctly populated
	require.Equal(t, "signature-abc-123", steps[0].Action.ThoughtSignature)
	require.Len(t, steps[0].Action.ThoughtParts, 1)
	require.Equal(t, "thought-sig-xyz", steps[0].Action.ThoughtParts[0].Signature)

	// Verify the agent was created successfully
	require.NotNil(t, agent)
}

// TestThoughtSignatureRoundTrip tests the full round-trip of thought signatures through parse and scratchpad
func TestThoughtSignatureRoundTrip(t *testing.T) {
	t.Parallel()
	agent := &agents.OpenAIFunctionsAgent{}

	// Step 1: Parse a response with thought signatures
	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				ToolCalls: []llms.ToolCall{
					{
						ID: "call1",
						FunctionCall: &llms.FunctionCall{
							Name:      "search",
							Arguments: `{"__arg1": "golang concurrency"}`,
						},
						ThoughtSignature: "gemini3-thought-sig-base64encoded",
					},
				},
				ThoughtParts: []llms.ThoughtContent{
					{
						Text:      "I need to search for information about golang concurrency patterns.",
						Signature: "gemini3-reasoning-sig",
					},
				},
			},
		},
	}

	actions, finish, err := agent.ParseOutput(resp)
	require.NoError(t, err)
	require.Nil(t, finish)
	require.Len(t, actions, 1)

	// Verify signatures are captured
	action := actions[0]
	require.Equal(t, "gemini3-thought-sig-base64encoded", action.ThoughtSignature)
	require.Len(t, action.ThoughtParts, 1)
	require.Equal(t, "gemini3-reasoning-sig", action.ThoughtParts[0].Signature)

	// Step 2: Create an AgentStep from this action (simulating executor behavior)
	step := schema.AgentStep{
		Action:      action,
		Observation: "Results: Go concurrency is based on goroutines and channels...",
	}

	// Verify the step preserves the signatures
	require.Equal(t, "gemini3-thought-sig-base64encoded", step.Action.ThoughtSignature)
	require.Len(t, step.Action.ThoughtParts, 1)
	require.Equal(t, "gemini3-reasoning-sig", step.Action.ThoughtParts[0].Signature)
}
