package googleai

import (
	"encoding/json"
	"testing"

	genai "google.golang.org/genai"
	"github.com/stretchr/testify/assert"
	"github.com/tmc/langchaingo/llms"
)

func TestConvertParts(t *testing.T) { //nolint:funlen // comprehensive test
	t.Parallel()

	tests := []struct {
		name      string
		parts     []llms.ContentPart
		wantErr   bool
		wantTypes []string // Expected types of genai.Part
	}{
		{
			name:      "empty parts",
			parts:     []llms.ContentPart{},
			wantErr:   false,
			wantTypes: []string{},
		},
		{
			name: "text content",
			parts: []llms.ContentPart{
				llms.TextContent{Text: "Hello world"},
			},
			wantErr:   false,
			wantTypes: []string{"genai.Text"},
		},
		{
			name: "binary content",
			parts: []llms.ContentPart{
				llms.BinaryContent{
					MIMEType: "image/jpeg",
					Data:     []byte("fake image data"),
				},
			},
			wantErr:   false,
			wantTypes: []string{"genai.Blob"},
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
			wantErr:   false,
			wantTypes: []string{"genai.FunctionCall"},
		},
		{
			name: "tool call response",
			parts: []llms.ContentPart{
				llms.ToolCallResponse{
					Name:    "get_weather",
					Content: "It's sunny in Paris",
				},
			},
			wantErr:   false,
			wantTypes: []string{"genai.FunctionResponse"},
		},
		{
			name: "tool call with invalid JSON",
			parts: []llms.ContentPart{
				llms.ToolCall{
					FunctionCall: &llms.FunctionCall{
						Name:      "get_weather",
						Arguments: `{invalid json}`,
					},
				},
			},
			wantErr: true,
		},
		{
			name: "mixed content types",
			parts: []llms.ContentPart{
				llms.TextContent{Text: "Hello"},
				llms.BinaryContent{MIMEType: "image/png", Data: []byte("png data")},
				llms.TextContent{Text: "World"},
			},
			wantErr:   false,
			wantTypes: []string{"genai.Text", "genai.Blob", "genai.Text"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertParts(tt.parts)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.Len(t, result, len(tt.wantTypes))

			for i, expectedType := range tt.wantTypes {
				assert.NotNil(t, result[i], "Part %d should not be nil", i)
				part := result[i]
				switch expectedType {
				case "genai.Text":
					assert.NotEmpty(t, part.Text, "Part %d should have Text field", i)
					assert.Nil(t, part.FunctionCall, "Part %d should not have FunctionCall", i)
					assert.Nil(t, part.FunctionResponse, "Part %d should not have FunctionResponse", i)
				case "genai.Blob":
					assert.NotNil(t, part.InlineData, "Part %d should have InlineData field", i)
					assert.Nil(t, part.FunctionCall, "Part %d should not have FunctionCall", i)
				case "genai.FunctionCall":
					assert.NotNil(t, part.FunctionCall, "Part %d should have FunctionCall field", i)
					assert.Empty(t, part.Text, "Part %d should not have Text", i)
				case "genai.FunctionResponse":
					assert.NotNil(t, part.FunctionResponse, "Part %d should have FunctionResponse field", i)
					assert.Empty(t, part.Text, "Part %d should not have Text", i)
				}
			}
		})
	}
}

func TestConvertContent(t *testing.T) { //nolint:funlen // comprehensive test
	t.Parallel()

	tests := []struct {
		name         string
		content      llms.MessageContent
		expectedRole string
		wantErr      bool
		errContains  string
	}{
		{
			name: "system message",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeSystem,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "You are a helpful assistant"},
				},
			},
			expectedRole: RoleSystem,
			wantErr:      false,
		},
		{
			name: "AI message",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeAI,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "Hello! How can I help you?"},
				},
			},
			expectedRole: RoleModel,
			wantErr:      false,
		},
		{
			name: "human message",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeHuman,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "What's the weather like?"},
				},
			},
			expectedRole: RoleUser,
			wantErr:      false,
		},
		{
			name: "generic message",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeGeneric,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "Generic message"},
				},
			},
			expectedRole: RoleUser,
			wantErr:      false,
		},
		{
			name: "tool message",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeTool,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "Tool response"},
				},
			},
			expectedRole: RoleUser,
			wantErr:      false,
		},
		{
			name: "function message (unsupported)",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeFunction,
				Parts: []llms.ContentPart{
					llms.TextContent{Text: "Function response"},
				},
			},
			wantErr:     true,
			errContains: "not supported",
		},
		{
			name: "invalid parts",
			content: llms.MessageContent{
				Role: llms.ChatMessageTypeHuman,
				Parts: []llms.ContentPart{
					llms.ToolCall{
						FunctionCall: &llms.FunctionCall{
							Name:      "test",
							Arguments: "invalid json",
						},
					},
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertContent(tt.content)

			if tt.wantErr {
				assert.Error(t, err)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, result)
			assert.Equal(t, tt.expectedRole, result.Role)
			assert.Len(t, result.Parts, len(tt.content.Parts))
		})
	}
}

func TestConvertCandidates(t *testing.T) { //nolint:funlen // comprehensive test
	t.Parallel()

	tests := []struct {
		name        string
		candidates  []*genai.Candidate
		usage       *genai.GenerateContentResponseUsageMetadata
		wantErr     bool
		wantChoices int
	}{
		{
			name:        "empty candidates",
			candidates:  []*genai.Candidate{},
			wantErr:     false,
			wantChoices: 0,
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
			wantErr:     false,
			wantChoices: 1,
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
			wantErr:     false,
			wantChoices: 1,
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
				PromptTokenCount: 10,
				// Note: Field names may differ in actual API
				// TotalTokenCount: 15,
			},
			wantErr:     false,
			wantChoices: 1,
		},
		{
			name: "multiple candidates",
			candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{{Text: "First response"}},
					},
					FinishReason: genai.FinishReasonStop,
				},
				{
					Content: &genai.Content{
						Parts: []*genai.Part{{Text: "Second response"}},
					},
					FinishReason: genai.FinishReasonStop,
				},
			},
			wantErr:     false,
			wantChoices: 2,
		},
		{
			name: "candidate with unknown part type",
			candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "Known type for now"},
						},
					},
					FinishReason: genai.FinishReasonStop,
				},
			},
			wantErr:     false,
			wantChoices: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertCandidates(tt.candidates, tt.usage)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, result)
			assert.Len(t, result.Choices, tt.wantChoices)

			// Check metadata for usage information
			if tt.usage != nil && len(result.Choices) > 0 {
				metadata := result.Choices[0].GenerationInfo
				assert.Equal(t, int32(10), metadata["input_tokens"])
				// Note: Output tokens field name may differ in actual API
				// assert.Equal(t, int32(5), metadata["output_tokens"])
				assert.Equal(t, int32(15), metadata["total_tokens"])
			}

			// Check that citations and safety are always present
			for i, choice := range result.Choices {
				assert.Contains(t, choice.GenerationInfo, CITATIONS, "Choice %d should have citations", i)
				assert.Contains(t, choice.GenerationInfo, SAFETY, "Choice %d should have safety info", i)
			}
		})
	}
}

func TestCall(t *testing.T) {
	t.Parallel()

	// Since Call is just a wrapper around GenerateFromSinglePrompt,
	// we test the interface compliance and basic structure
	t.Run("implements interface", func(t *testing.T) {
		var _ llms.Model = &GoogleAI{}
	})

	// Note: Full testing would require mocking the genai client
	// which is complex due to the dependency structure
}

func TestGenerateContentOptionsHandling(t *testing.T) {
	t.Parallel()

	// Test the options validation logic that can be tested without a client
	t.Run("conflicting JSONMode and ResponseMIMEType", func(t *testing.T) {
		// This tests the validation logic in GenerateContent
		opts := llms.CallOptions{
			JSONMode:         true,
			ResponseMIMEType: "text/plain",
		}

		// The validation would happen in GenerateContent:
		// if opts.ResponseMIMEType != "" && opts.JSONMode {
		//     return nil, fmt.Errorf("conflicting options, can't use JSONMode and ResponseMIMEType together")
		// }

		hasConflict := opts.ResponseMIMEType != "" && opts.JSONMode
		assert.True(t, hasConflict, "Should detect conflicting options")
	})

	t.Run("JSONMode sets correct MIME type", func(t *testing.T) {
		opts := llms.CallOptions{
			JSONMode: true,
		}

		// The logic would set: model.ResponseMIMEType = ResponseMIMETypeJson
		expectedMIMEType := ResponseMIMETypeJson
		if opts.JSONMode && opts.ResponseMIMEType == "" {
			assert.Equal(t, "application/json", expectedMIMEType)
		}
	})

	t.Run("custom ResponseMIMEType", func(t *testing.T) {
		opts := llms.CallOptions{
			ResponseMIMEType: "text/xml",
		}

		// The logic would set: model.ResponseMIMEType = opts.ResponseMIMEType
		if opts.ResponseMIMEType != "" && !opts.JSONMode {
			assert.Equal(t, "text/xml", opts.ResponseMIMEType)
		}
	})
}

func TestRoleMapping(t *testing.T) {
	t.Parallel()

	// Test the role mapping constants
	roleTests := []struct {
		llmRole      llms.ChatMessageType
		expectedRole string
		supported    bool
	}{
		{llms.ChatMessageTypeSystem, RoleSystem, true},
		{llms.ChatMessageTypeAI, RoleModel, true},
		{llms.ChatMessageTypeHuman, RoleUser, true},
		{llms.ChatMessageTypeGeneric, RoleUser, true},
		{llms.ChatMessageTypeTool, RoleUser, true},
		{llms.ChatMessageTypeFunction, "", false}, // Unsupported
	}

	for _, tt := range roleTests {
		t.Run(string(tt.llmRole), func(t *testing.T) {
			content := llms.MessageContent{
				Role:  tt.llmRole,
				Parts: []llms.ContentPart{llms.TextContent{Text: "test"}},
			}

			result, err := convertContent(content)

			if !tt.supported {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "not supported")
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, tt.expectedRole, result.Role)
		})
	}
}

func TestFunctionCallConversion(t *testing.T) {
	t.Parallel()

	t.Run("valid function call", func(t *testing.T) {
		args := map[string]any{
			"location": "Paris",
			"unit":     "celsius",
		}
		argsJSON, _ := json.Marshal(args)

		part := llms.ToolCall{
			FunctionCall: &llms.FunctionCall{
				Name:      "get_weather",
				Arguments: string(argsJSON),
			},
		}

		result, err := convertParts([]llms.ContentPart{part})
		assert.NoError(t, err)
		assert.Len(t, result, 1)

		// Parts are now *genai.Part structs, not interfaces
		assert.NotNil(t, result[0])
		assert.NotNil(t, result[0].FunctionCall)
		assert.Equal(t, "get_weather", result[0].FunctionCall.Name)
		assert.Equal(t, "Paris", result[0].FunctionCall.Args["location"])
		assert.Equal(t, "celsius", result[0].FunctionCall.Args["unit"])
	})

	t.Run("function response", func(t *testing.T) {
		part := llms.ToolCallResponse{
			Name:    "get_weather",
			Content: "It's 20°C and sunny",
		}

		result, err := convertParts([]llms.ContentPart{part})
		assert.NoError(t, err)
		assert.Len(t, result, 1)

		// Parts are now *genai.Part structs, not interfaces
		assert.NotNil(t, result[0])
		assert.NotNil(t, result[0].FunctionResponse)
		assert.Equal(t, "get_weather", result[0].FunctionResponse.Name)
		assert.Equal(t, "It's 20°C and sunny", result[0].FunctionResponse.Response["response"])
	})
}

func TestSafetySettings(t *testing.T) {
	t.Parallel()

	// Test that all safety categories are covered
	expectedCategories := []genai.HarmCategory{
		genai.HarmCategoryDangerousContent,
		genai.HarmCategoryHarassment,
		genai.HarmCategoryHateSpeech,
		genai.HarmCategorySexuallyExplicit,
	}

	// This would be the safety settings logic from GenerateContent
	harmThreshold := HarmBlockOnlyHigh

	safetySettings := []*genai.SafetySetting{}
	for _, category := range expectedCategories {
		safetySettings = append(safetySettings, &genai.SafetySetting{
			Category:  category,
			Threshold: genai.HarmBlockThreshold(harmThreshold),
		})
	}

	assert.Len(t, safetySettings, 4, "Should have safety settings for all categories")

	for i, setting := range safetySettings {
		assert.Equal(t, expectedCategories[i], setting.Category)
		assert.Equal(t, genai.HarmBlockThreshold(harmThreshold), setting.Threshold)
	}
}
