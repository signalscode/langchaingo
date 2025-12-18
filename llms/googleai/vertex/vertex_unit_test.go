package vertex

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"google.golang.org/genai"
)

func TestConvertToolSchemaType(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected genai.Type
	}{
		{"object type", "object", genai.TypeObject},
		{"string type", "string", genai.TypeString},
		{"number type", "number", genai.TypeNumber},
		{"integer type", "integer", genai.TypeInteger},
		{"boolean type", "boolean", genai.TypeBoolean},
		{"array type", "array", genai.TypeArray},
		{"unknown type", "unknown", genai.TypeUnspecified},
		{"empty type", "", genai.TypeUnspecified},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertToolSchemaType(tt.input)
			if result != tt.expected {
				t.Errorf("convertToolSchemaType(%q) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

func TestConvertParts(t *testing.T) { //nolint:funlen // comprehensive test
	tests := []struct {
		name    string
		parts   []llms.ContentPart
		wantErr bool
		check   func(t *testing.T, result []*genai.Part)
	}{
		{
			name: "text content",
			parts: []llms.ContentPart{
				llms.TextContent{Text: "Hello, world!"},
			},
			wantErr: false,
			check: func(t *testing.T, result []*genai.Part) {
				if len(result) != 1 {
					t.Fatalf("expected 1 part, got %d", len(result))
				}
				if result[0].Text != "Hello, world!" {
					t.Errorf("expected text 'Hello, world!', got %q", result[0].Text)
				}
			},
		},
		{
			name: "binary content",
			parts: []llms.ContentPart{
				llms.BinaryContent{
					MIMEType: "image/png",
					Data:     []byte("fake image data"),
				},
			},
			wantErr: false,
			check: func(t *testing.T, result []*genai.Part) {
				if len(result) != 1 {
					t.Fatalf("expected 1 part, got %d", len(result))
				}
				if result[0].InlineData == nil {
					t.Fatal("expected InlineData to be set")
				}
				if result[0].InlineData.MIMEType != "image/png" {
					t.Errorf("expected MIME type 'image/png', got %q", result[0].InlineData.MIMEType)
				}
			},
		},
		{
			name: "tool call",
			parts: []llms.ContentPart{
				llms.ToolCall{
					FunctionCall: &llms.FunctionCall{
						Name:      "get_weather",
						Arguments: `{"location": "Paris"}`,
					},
				},
			},
			wantErr: false,
			check: func(t *testing.T, result []*genai.Part) {
				if len(result) != 1 {
					t.Fatalf("expected 1 part, got %d", len(result))
				}
				if result[0].FunctionCall == nil {
					t.Fatal("expected FunctionCall to be set")
				}
				if result[0].FunctionCall.Name != "get_weather" {
					t.Errorf("expected name 'get_weather', got %q", result[0].FunctionCall.Name)
				}
			},
		},
		{
			name: "tool call response",
			parts: []llms.ContentPart{
				llms.ToolCallResponse{
					Name:    "get_weather",
					Content: "It's sunny",
				},
			},
			wantErr: false,
			check: func(t *testing.T, result []*genai.Part) {
				if len(result) != 1 {
					t.Fatalf("expected 1 part, got %d", len(result))
				}
				if result[0].FunctionResponse == nil {
					t.Fatal("expected FunctionResponse to be set")
				}
				if result[0].FunctionResponse.Name != "get_weather" {
					t.Errorf("expected name 'get_weather', got %q", result[0].FunctionResponse.Name)
				}
			},
		},
		{
			name: "tool call with invalid JSON",
			parts: []llms.ContentPart{
				llms.ToolCall{
					FunctionCall: &llms.FunctionCall{
						Name:      "test",
						Arguments: "not valid json",
					},
				},
			},
			wantErr: true,
			check:   nil,
		},
		{
			name:    "empty parts",
			parts:   []llms.ContentPart{},
			wantErr: false,
			check: func(t *testing.T, result []*genai.Part) {
				if len(result) != 0 {
					t.Errorf("expected 0 parts, got %d", len(result))
				}
			},
		},
		{
			name: "thought content",
			parts: []llms.ContentPart{
				llms.ThoughtContent{
					Text: "Let me think...",
				},
			},
			wantErr: false,
			check: func(t *testing.T, result []*genai.Part) {
				if len(result) != 1 {
					t.Fatalf("expected 1 part, got %d", len(result))
				}
				if !result[0].Thought {
					t.Error("expected Thought to be true")
				}
				if result[0].Text != "Let me think..." {
					t.Errorf("expected text 'Let me think...', got %q", result[0].Text)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertParts(tt.parts)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.check != nil {
				tt.check(t, result)
			}
		})
	}
}

func TestConvertContent(t *testing.T) {
	tests := []struct {
		name        string
		content     llms.MessageContent
		wantRole    string
		wantErr     bool
		errContains string
	}{
		{
			name: "human message",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeHuman,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "Hello"},
				},
			},
			wantRole: RoleUser,
			wantErr:  false,
		},
		{
			name: "AI message",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeAI,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "Hi there"},
				},
			},
			wantRole: RoleModel,
			wantErr:  false,
		},
		{
			name: "system message",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeSystem,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "You are helpful"},
				},
			},
			wantRole: RoleSystem,
			wantErr:  false,
		},
		{
			name: "function message (unsupported)",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeFunction,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "Result"},
				},
			},
			wantErr:     true,
			errContains: "not supported",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertContent(tt.content)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("expected error containing %q, got %q", tt.errContains, err.Error())
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if result.Role != tt.wantRole {
				t.Errorf("expected role %q, got %q", tt.wantRole, result.Role)
			}
		})
	}
}

func TestConvertCandidates(t *testing.T) { //nolint:funlen // comprehensive test
	tests := []struct {
		name        string
		candidates  []*genai.Candidate
		usage       *genai.GenerateContentResponseUsageMetadata
		wantChoices int
		wantErr     bool
	}{
		{
			name:        "empty candidates",
			candidates:  []*genai.Candidate{},
			wantChoices: 0,
			wantErr:     false,
		},
		{
			name: "single text candidate",
			candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "Hello world"},
						},
					},
					FinishReason: genai.FinishReasonStop,
				},
			},
			wantChoices: 1,
			wantErr:     false,
		},
		{
			name: "candidate with function call",
			candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{
								FunctionCall: &genai.FunctionCall{
									Name: "get_weather",
									Args: map[string]any{"location": "Paris"},
								},
							},
						},
					},
					FinishReason: genai.FinishReasonStop,
				},
			},
			wantChoices: 1,
			wantErr:     false,
		},
		{
			name: "candidate with usage metadata",
			candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "Response with usage"},
						},
					},
					FinishReason: genai.FinishReasonStop,
				},
			},
			usage: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     10,
				CandidatesTokenCount: 5,
				TotalTokenCount:      15,
			},
			wantChoices: 1,
			wantErr:     false,
		},
		{
			name: "candidate with thought content",
			candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{
								Text:    "Let me think...",
								Thought: true,
							},
							{Text: "The answer is 42"},
						},
					},
					FinishReason: genai.FinishReasonStop,
				},
			},
			wantChoices: 1,
			wantErr:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertCandidates(tt.candidates, tt.usage)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(result.Choices) != tt.wantChoices {
				t.Errorf("expected %d choices, got %d", tt.wantChoices, len(result.Choices))
			}

			// Verify usage metadata is properly extracted
			if tt.usage != nil && len(result.Choices) > 0 {
				metadata := result.Choices[0].GenerationInfo
				if metadata["input_tokens"] != int32(10) {
					t.Errorf("expected input_tokens=10, got %v", metadata["input_tokens"])
				}
				if metadata["output_tokens"] != int32(5) {
					t.Errorf("expected output_tokens=5, got %v", metadata["output_tokens"])
				}
				if metadata["total_tokens"] != int32(15) {
					t.Errorf("expected total_tokens=15, got %v", metadata["total_tokens"])
				}
			}
		})
	}
}

func TestConvertTools(t *testing.T) { //nolint:funlen // comprehensive test
	tests := []struct {
		name    string
		tools   []llms.Tool
		wantErr bool
		check   func(t *testing.T, result []*genai.Tool)
	}{
		{
			name:    "empty tools",
			tools:   nil,
			wantErr: false,
			check: func(t *testing.T, result []*genai.Tool) {
				if result != nil {
					t.Errorf("expected nil, got %v", result)
				}
			},
		},
		{
			name: "unsupported tool type",
			tools: []llms.Tool{
				{Type: "unsupported"},
			},
			wantErr: true,
			check:   nil,
		},
		{
			name: "valid function tool",
			tools: []llms.Tool{
				{
					Type: "function",
					Function: &llms.FunctionDefinition{
						Name:        "get_weather",
						Description: "Get the weather",
						Parameters: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"location": map[string]any{
									"type":        "string",
									"description": "City name",
								},
							},
							"required": []string{"location"},
						},
					},
				},
			},
			wantErr: false,
			check: func(t *testing.T, result []*genai.Tool) {
				if len(result) != 1 {
					t.Fatalf("expected 1 tool, got %d", len(result))
				}
				if len(result[0].FunctionDeclarations) != 1 {
					t.Fatalf("expected 1 function declaration, got %d", len(result[0].FunctionDeclarations))
				}
				fd := result[0].FunctionDeclarations[0]
				if fd.Name != "get_weather" {
					t.Errorf("expected name 'get_weather', got %q", fd.Name)
				}
				if fd.Description != "Get the weather" {
					t.Errorf("expected description 'Get the weather', got %q", fd.Description)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertTools(tt.tools)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.check != nil {
				tt.check(t, result)
			}
		})
	}
}

func TestConstants(t *testing.T) {
	// Verify role constants are correct
	if RoleSystem != "system" {
		t.Errorf("expected RoleSystem='system', got %q", RoleSystem)
	}
	if RoleModel != "model" {
		t.Errorf("expected RoleModel='model', got %q", RoleModel)
	}
	if RoleUser != "user" {
		t.Errorf("expected RoleUser='user', got %q", RoleUser)
	}
	if RoleTool != "tool" {
		t.Errorf("expected RoleTool='tool', got %q", RoleTool)
	}
	if ResponseMIMETypeJson != "application/json" {
		t.Errorf("expected ResponseMIMETypeJson='application/json', got %q", ResponseMIMETypeJson)
	}
}

func TestShowContent(t *testing.T) {
	// Test the debug helper function
	var buf strings.Builder
	contents := []*genai.Content{
		{
			Role: "user",
			Parts: []*genai.Part{
				{Text: "Hello"},
			},
		},
		{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "Hi there!"},
			},
		},
	}

	showContent(&buf, contents)
	output := buf.String()

	if !strings.Contains(output, "Content (len=2)") {
		t.Error("expected output to contain 'Content (len=2)'")
	}
	if !strings.Contains(output, "Role=user") {
		t.Error("expected output to contain 'Role=user'")
	}
	if !strings.Contains(output, "Role=model") {
		t.Error("expected output to contain 'Role=model'")
	}
}

// Ensure JSON marshaling works correctly with our types
func TestJSONMarshaling(t *testing.T) {
	funcCall := llms.FunctionCall{
		Name:      "test_func",
		Arguments: `{"key": "value"}`,
	}

	data, err := json.Marshal(funcCall)
	if err != nil {
		t.Fatalf("failed to marshal FunctionCall: %v", err)
	}

	var unmarshaled llms.FunctionCall
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("failed to unmarshal FunctionCall: %v", err)
	}

	if unmarshaled.Name != funcCall.Name {
		t.Errorf("name mismatch: expected %q, got %q", funcCall.Name, unmarshaled.Name)
	}
	if unmarshaled.Arguments != funcCall.Arguments {
		t.Errorf("arguments mismatch: expected %q, got %q", funcCall.Arguments, unmarshaled.Arguments)
	}
}

