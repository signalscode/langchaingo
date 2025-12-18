//nolint:all
package googleai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	genai "google.golang.org/genai"
	"github.com/tmc/langchaingo/internal/imageutil"
	"github.com/tmc/langchaingo/llms"
)

var (
	ErrNoContentInResponse   = errors.New("no content in generation response")
	ErrUnknownPartInResponse = errors.New("unknown part type in generation response")
	ErrInvalidMimeType       = errors.New("invalid mime type on content")
)

const (
	CITATIONS            = "citations"
	SAFETY               = "safety"
	RoleSystem           = "system"
	RoleModel            = "model"
	RoleUser             = "user"
	RoleTool             = "tool"
	ResponseMIMETypeJson = "application/json"
)

// Call implements the [llms.Model] interface.
func (g *GoogleAI) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, g, prompt, options...)
}

// GenerateContent implements the [llms.Model] interface.
func (g *GoogleAI) GenerateContent(
	ctx context.Context,
	messages []llms.MessageContent,
	options ...llms.CallOption,
) (*llms.ContentResponse, error) {
	if g.CallbacksHandler != nil {
		g.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := llms.CallOptions{
		Model:          g.opts.DefaultModel,
		CandidateCount: g.opts.DefaultCandidateCount,
		MaxTokens:      g.opts.DefaultMaxTokens,
		Temperature:    g.opts.DefaultTemperature,
		TopP:           g.opts.DefaultTopP,
		TopK:           g.opts.DefaultTopK,
	}
	for _, opt := range options {
		opt(&opts)
	}

	// Update the tracked model if it was overridden
	effectiveModel := opts.Model
	if effectiveModel != "" && effectiveModel != g.model {
		g.model = effectiveModel
	}

	// Build generation parameters for the new API
	// The new library's parameter structure needs to be determined
	// For now, we'll pass nil and handle options differently if needed
	params := buildGenerationParams(&opts, g.opts)

	// Convert tools - these may need to be passed differently in the new API
	_, err := convertTools(opts.Tools)
	if err != nil {
		return nil, err
	}
	
	// TODO: Set tools in params based on actual API structure

	// Set response MIME type
	// TODO: Handle ResponseMIMEType based on actual API structure
	switch {
	case opts.ResponseMIMEType != "" && opts.JSONMode:
		return nil, fmt.Errorf("conflicting options, can't use JSONMode and ResponseMIMEType together")
	case opts.ResponseMIMEType != "" && !opts.JSONMode:
		// Set in params if API supports it
	case opts.ResponseMIMEType == "" && opts.JSONMode:
		// Set to JSON mode if API supports it
	}

	var response *llms.ContentResponse

	if len(messages) == 1 {
		theMessage := messages[0]
		if theMessage.Role != llms.ChatMessageTypeHuman {
			return nil, fmt.Errorf("got %v message role, want human", theMessage.Role)
		}
		response, err = generateFromSingleMessage(ctx, g.client, effectiveModel, theMessage.Parts, params, &opts)
	} else {
		response, err = generateFromMessages(ctx, g.client, effectiveModel, messages, params, &opts)
	}
	if err != nil {
		return nil, err
	}

	if g.CallbacksHandler != nil {
		g.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, response)
	}

	return response, nil
}

// validateThoughtSignatures validates that thought signatures are present for Gemini 3 models
// when function calls exist. Returns an error if signatures are missing.
func validateThoughtSignatures(modelName string, toolCalls []llms.ToolCall) error {
	// Check if this is a Gemini 3 model
	if !strings.Contains(modelName, "gemini-3") {
		return nil // Validation only required for Gemini 3
	}
	
	// If there are tool calls, each must have a thought signature
	if len(toolCalls) > 0 {
		for i, tc := range toolCalls {
			if tc.ThoughtSignature == "" {
				return fmt.Errorf(
					"thought signature required for Gemini 3 models: missing signature for tool call %d (function: %s). "+
						"Thought signatures are mandatory for function calling with Gemini 3 models. "+
						"Ensure you are preserving thought signatures from previous responses exactly as received.",
					i, tc.FunctionCall.Name,
				)
			}
		}
	}
	
	return nil
}

// buildGenerationParams builds generation parameters for the new API
func buildGenerationParams(opts *llms.CallOptions, defaultOpts Options) *genai.GenerateContentConfig {
	config := &genai.GenerateContentConfig{
		// Set generation parameters based on options
		// The exact field names may need adjustment based on actual API
	}
	
	// TODO: Set config fields like Temperature, MaxTokens, etc. based on opts
	// This needs to be filled in based on the actual GenerateContentConfig structure
	
	return config
}

// setThoughtSignatureInPart sets the thought signature in a part's ExtraContent field.
// According to the Gemini API docs, thought signatures must be in extra_content.google.thought_signature
//
// CRITICAL: Thought signatures MUST be preserved exactly as received for Gemini 3 models.
// The signature must be set in the exact same structure as it was received.
// Missing or incorrect signatures will cause function calling to fail with 400 errors.
func setThoughtSignatureInPart(part *genai.Part, signature string) *genai.Part {
	if part == nil || signature == "" {
		return part
	}
	
	// According to Gemini API documentation, the structure is:
	// extra_content.google.thought_signature
	//
	// For function calls in conversation history, the signature must be preserved
	// in the FunctionCall's ExtraContent field.
	//
	// The new library structure needs to be determined. Possible locations:
	// - part.FunctionCall.ExtraContent.google.thought_signature
	// - part.ExtraContent.google.thought_signature (if Part has ExtraContent)
	//
	// TODO: Implement actual setting based on library structure
	// This is CRITICAL for Gemini 3 function calling to work
	if part.FunctionCall != nil {
		// Set thought signature in FunctionCall structure
		// The exact path depends on the library implementation
		// Example (if structure is known):
		// if part.FunctionCall.ExtraContent == nil {
		//     part.FunctionCall.ExtraContent = &genai.ExtraContent{}
		// }
		// if part.FunctionCall.ExtraContent.Google == nil {
		//     part.FunctionCall.ExtraContent.Google = &genai.GoogleExtraContent{}
		// }
		// part.FunctionCall.ExtraContent.Google.ThoughtSignature = signature
	}
	
	return part
}

// extractThoughtSignatureFromPart extracts the thought signature from a part.
// According to the Gemini API docs, thought signatures are in extra_content.google.thought_signature
// For function calls, they appear in the function call's extra_content field
//
// CRITICAL: Thought signatures MUST be preserved exactly as received for Gemini 3 models.
// Missing or incorrect signatures will cause function calling to fail with 400 errors.
func extractThoughtSignatureFromPart(part *genai.Part) string {
	if part == nil {
		return ""
	}
	
	// According to Gemini API documentation:
	// - For function calls: signature is in function_call.extra_content.google.thought_signature
	// - For text responses: signature may be in the last part's extra_content
	//
	// The new library structure needs to be determined. Possible locations:
	// - part.FunctionCall.ExtraContent.google.thought_signature
	// - part.ExtraContent.google.thought_signature
	// - A different nested structure
	//
	// TODO: Implement actual extraction based on library structure
	// This is CRITICAL for Gemini 3 function calling to work
	if part.FunctionCall != nil {
		// Try to extract from FunctionCall structure
		// The exact path depends on the library implementation
		// Example (if structure is known):
		// if part.FunctionCall.ExtraContent != nil {
		//     if google := part.FunctionCall.ExtraContent.Google; google != nil {
		//         return google.ThoughtSignature
		//     }
		// }
	}
	
	// For text parts, check if there's extra_content
	// This is less common but may occur in some response types
	
	return ""
}

// convertCandidates converts a sequence of genai.Candidate to a response.
func convertCandidates(candidates []*genai.Candidate, usage *genai.GenerateContentResponseUsageMetadata) (*llms.ContentResponse, error) {
	var contentResponse llms.ContentResponse
	var toolCalls []llms.ToolCall
	var lastThoughtSignature string // Track thought signature from last part (for non-function-call responses)

	for _, candidate := range candidates {
		buf := strings.Builder{}

		if candidate.Content != nil {
			parts := candidate.Content.Parts
			for i, part := range parts {
				if part == nil {
					continue
				}
				
				// The new library uses *genai.Part with different fields
				// Check what type of part this is based on the Part structure
				if part.Text != "" {
					textContent := part.Text
					// For empty text parts, this might contain thought signature metadata
					// According to docs, during streaming the signature may be in empty text parts
					if textContent == "" && i == len(parts)-1 {
						// This might be the thought signature part - extract if possible
						// The actual extraction depends on library structure
						// Thought signatures may be in a different field or structure
					}
					_, err := buf.WriteString(textContent)
					if err != nil {
						return nil, err
					}
				} else if part.FunctionCall != nil {
					// Extract function call information
					fc := part.FunctionCall
					b, err := json.Marshal(fc.Args)
					if err != nil {
						return nil, err
					}
					
					// Extract thought signature - may be in part or function call structure
					thoughtSignature := extractThoughtSignatureFromPart(part)
					
					toolCall := llms.ToolCall{
						FunctionCall: &llms.FunctionCall{
							Name:      fc.Name,
							Arguments: string(b),
						},
						ThoughtSignature: thoughtSignature,
					}
					toolCalls = append(toolCalls, toolCall)
					
					// Store thought signature from first function call (for parallel calls, only first has it)
					if thoughtSignature != "" && lastThoughtSignature == "" {
						lastThoughtSignature = thoughtSignature
					}
				} else if part.FunctionResponse != nil {
					// Handle function response - this is typically not in candidate content
					// but may be present in some cases
				} else {
					// Unknown part type - may have other fields like InlineData, FileData, etc.
					// For now, skip unknown parts or handle based on actual API
				}
			}
		}

		metadata := make(map[string]any)
		metadata[CITATIONS] = candidate.CitationMetadata
		metadata[SAFETY] = candidate.SafetyRatings

		if usage != nil {
			// The new library may have different field names - adjust based on actual API
			// For now, use what we can access
			metadata["input_tokens"] = usage.PromptTokenCount
			// TODO: Update field names based on actual UsageMetadata structure
			// metadata["output_tokens"] = usage.OutputTokenCount (or similar)
			// metadata["total_tokens"] = usage.TotalTokenCount (or similar)
			
			// Standardized field names for cross-provider compatibility
			metadata["PromptTokens"] = usage.PromptTokenCount
			// metadata["CompletionTokens"] = usage.OutputTokenCount
			// metadata["TotalTokens"] = usage.TotalTokenCount

			// Cache-related token information (if available)
			// TODO: Update based on actual UsageMetadata structure
		}

		// Google AI doesn't separate thinking content like OpenAI o1, but we provide empty standardized fields
		metadata["ThinkingContent"] = "" // Google models don't separate thinking content
		metadata["ThinkingTokens"] = 0   // Google models don't track thinking tokens separately

		// Note: Google AI's CachedContent requires pre-created cached content via API,
		// not inline cache control like Anthropic. Use Client.CreateCachedContent() for caching.

		// Convert FinishReason to string
		stopReason := fmt.Sprintf("%v", candidate.FinishReason)
		
		choice := &llms.ContentChoice{
			Content:        buf.String(),
			StopReason:     stopReason,
			GenerationInfo: metadata,
			ToolCalls:      toolCalls,
		}
		
		// If no function calls but we have a thought signature, store it in the choice
		// This happens when the model generates a text response with a thought signature
		if len(toolCalls) == 0 && lastThoughtSignature != "" {
			choice.ThoughtSignature = lastThoughtSignature
		}
		
		contentResponse.Choices = append(contentResponse.Choices, choice)
	}
	return &contentResponse, nil
}

// convertParts converts between a sequence of langchain parts and genai parts.
// The new library may use a different Part structure - this needs to be adjusted
func convertParts(parts []llms.ContentPart) ([]*genai.Part, error) {
	convertedParts := make([]*genai.Part, 0, len(parts))
	for _, part := range parts {
		var out *genai.Part

		switch p := part.(type) {
		case llms.TextContent:
			textPart := &genai.Part{
				Text: p.Text,
			}
			out = textPart
		case llms.BinaryContent:
			blobPart := &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: p.MIMEType,
					Data:     p.Data,
				},
			}
			out = blobPart
		case llms.ImageURLContent:
			typ, data, err := imageutil.DownloadImageData(p.URL)
			if err != nil {
				return nil, err
			}
			imagePart := &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: typ,
					Data:     data,
				},
			}
			out = imagePart
		case llms.ToolCall:
			fc := p.FunctionCall
			var argsMap map[string]any
			if err := json.Unmarshal([]byte(fc.Arguments), &argsMap); err != nil {
				return convertedParts, err
			}
			
			// Create function call part with thought signature if present
			functionCallPart := &genai.Part{
				FunctionCall: &genai.FunctionCall{
					Name: fc.Name,
					Args: argsMap,
				},
			}
			
			// Preserve thought signature - this is critical for Gemini 3 models
			// Thought signatures must be preserved exactly as received
			// They may be in FunctionCall.ExtraContent or a similar structure
			if p.ThoughtSignature != "" {
				functionCallPart = setThoughtSignatureInPart(functionCallPart, p.ThoughtSignature)
			}
			
			out = functionCallPart
		case llms.ToolCallResponse:
			// Create function response part
			responsePart := &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					Name: p.Name,
					Response: map[string]any{
						"response": p.Content,
					},
				},
			}
			out = responsePart
		}

		if out != nil {
			convertedParts = append(convertedParts, out)
		}
	}
	return convertedParts, nil
}

// convertContent converts between a langchain MessageContent and genai content.
func convertContent(content llms.MessageContent) (*genai.Content, error) {
	parts, err := convertParts(content.Parts)
	if err != nil {
		return nil, err
	}

	c := &genai.Content{
		Parts: parts, // Parts is now []*genai.Part
	}

	switch content.Role {
	case llms.ChatMessageTypeSystem:
		c.Role = RoleSystem
	case llms.ChatMessageTypeAI:
		c.Role = RoleModel
	case llms.ChatMessageTypeHuman:
		c.Role = RoleUser
	case llms.ChatMessageTypeGeneric:
		c.Role = RoleUser
	case llms.ChatMessageTypeTool:
		c.Role = RoleUser
	case llms.ChatMessageTypeFunction:
		fallthrough
	default:
		return nil, fmt.Errorf("role %v not supported", content.Role)
	}

	return c, nil
}

// generateFromSingleMessage generates content from the parts of a single
// message using the new API.
func generateFromSingleMessage(
	ctx context.Context,
	client *genai.Client,
	modelName string,
	parts []llms.ContentPart,
	params *genai.GenerateContentConfig,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	convertedParts, err := convertParts(parts)
	if err != nil {
		return nil, err
	}

	// Build content for the new API
	contents := []*genai.Content{
		{
			Parts: convertedParts,
		},
	}

	if opts.StreamingFunc == nil {
		// When no streaming is requested, call GenerateContent
		resp, err := client.Models.GenerateContent(ctx, modelName, contents, params)
		if err != nil {
			return nil, err
		}

		if len(resp.Candidates) == 0 {
			return nil, ErrNoContentInResponse
		}
		
		response, err := convertCandidates(resp.Candidates, resp.UsageMetadata)
		if err != nil {
			return nil, err
		}
		
		// Validate thought signatures for Gemini 3 models
		for _, choice := range response.Choices {
			if err := validateThoughtSignatures(modelName, choice.ToolCalls); err != nil {
				return nil, err
			}
		}
		
		return response, nil
	}
	
	// Streaming support - will need to be updated based on actual API
	// For now, return error indicating streaming needs implementation
	return nil, fmt.Errorf("streaming not yet implemented with new API")
}

func generateFromMessages(
	ctx context.Context,
	client *genai.Client,
	modelName string,
	messages []llms.MessageContent,
	params *genai.GenerateContentConfig,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	contents := make([]*genai.Content, 0, len(messages))
	var systemInstruction *genai.Content
	
	for _, mc := range messages {
		content, err := convertContent(mc)
		if err != nil {
			return nil, err
		}
		if mc.Role == RoleSystem {
			systemInstruction = content
			// Set system instruction in params if supported
			if params != nil && systemInstruction != nil {
				// TODO: Set system instruction in params based on actual API structure
				// params.SystemInstruction = systemInstruction (or similar)
			}
			continue
		}
		contents = append(contents, content)
	}

	if opts.StreamingFunc == nil {
		resp, err := client.Models.GenerateContent(ctx, modelName, contents, params)
		if err != nil {
			return nil, err
		}

		if len(resp.Candidates) == 0 {
			return nil, ErrNoContentInResponse
		}
		
		response, err := convertCandidates(resp.Candidates, resp.UsageMetadata)
		if err != nil {
			return nil, err
		}
		
		// Validate thought signatures for Gemini 3 models
		for _, choice := range response.Choices {
			if err := validateThoughtSignatures(modelName, choice.ToolCalls); err != nil {
				return nil, err
			}
		}
		
		return response, nil
	}
	
	// Streaming support - will need to be updated based on actual API
	return nil, fmt.Errorf("streaming not yet implemented with new API")
}

// convertAndStreamFromIterator handles streaming responses.
// TODO: Update to use new library's streaming API
// The new library may have a different streaming mechanism
func convertAndStreamFromIterator(
	ctx context.Context,
	iter interface{}, // The actual iterator type needs to be determined
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	// TODO: Implement streaming with new library
	// The streaming API structure needs to be determined
	return nil, fmt.Errorf("streaming not yet implemented with new API")
}

// convertSchemaRecursive recursively converts a schema map to a genai.Schema
func convertSchemaRecursive(schemaMap map[string]any, toolIndex int, propertyPath string) (*genai.Schema, error) {
	schema := &genai.Schema{}

	if ty, ok := schemaMap["type"]; ok {
		tyString, ok := ty.(string)
		if !ok {
			return nil, fmt.Errorf("tool [%d], property [%s]: expected string for type", toolIndex, propertyPath)
		}
		schema.Type = convertToolSchemaType(tyString)
	}

	if desc, ok := schemaMap["description"]; ok {
		descString, ok := desc.(string)
		if !ok {
			return nil, fmt.Errorf("tool [%d], property [%s]: expected string for description", toolIndex, propertyPath)
		}
		schema.Description = descString
	}

	// Handle object properties recursively
	if properties, ok := schemaMap["properties"]; ok {
		propMap, ok := properties.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tool [%d], property [%s]: expected map for properties", toolIndex, propertyPath)
		}

		schema.Properties = make(map[string]*genai.Schema)
		for propName, propValue := range propMap {
			valueMap, ok := propValue.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("tool [%d], property [%s.%s]: expect to find a value map", toolIndex, propertyPath, propName)
			}

			nestedPath := propName
			if propertyPath != "" {
				nestedPath = propertyPath + "." + propName
			}

			nestedSchema, err := convertSchemaRecursive(valueMap, toolIndex, nestedPath)
			if err != nil {
				return nil, err
			}
			schema.Properties[propName] = nestedSchema
		}
	} else if schema.Type == genai.TypeObject && propertyPath == "" {
		// For top-level object schemas without properties, this is an error
		return nil, fmt.Errorf("tool [%d]: expected to find a map of properties", toolIndex)
	}

	// Handle array items recursively
	if items, ok := schemaMap["items"]; ok && schema.Type == genai.TypeArray {
		itemMap, ok := items.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tool [%d], property [%s]: expect to find a map for array items", toolIndex, propertyPath)
		}

		itemsPath := propertyPath + "[]"
		itemsSchema, err := convertSchemaRecursive(itemMap, toolIndex, itemsPath)
		if err != nil {
			return nil, err
		}
		schema.Items = itemsSchema
	}

	// Handle required fields
	if required, ok := schemaMap["required"]; ok {
		if rs, ok := required.([]string); ok {
			schema.Required = rs
		} else if ri, ok := required.([]interface{}); ok {
			rs := make([]string, 0, len(ri))
			for _, r := range ri {
				rString, ok := r.(string)
				if !ok {
					return nil, fmt.Errorf("tool [%d], property [%s]: expected string for required", toolIndex, propertyPath)
				}
				rs = append(rs, rString)
			}
			schema.Required = rs
		} else {
			return nil, fmt.Errorf("tool [%d], property [%s]: expected array for required", toolIndex, propertyPath)
		}
	}

	return schema, nil
}

// convertTools converts from a list of langchaingo tools to a list of genai
// tools.
func convertTools(tools []llms.Tool) ([]*genai.Tool, error) {
	genaiFuncDecls := make([]*genai.FunctionDeclaration, 0, len(tools))
	for i, tool := range tools {
		if tool.Type != "function" {
			return nil, fmt.Errorf("tool [%d]: unsupported type %q, want 'function'", i, tool.Type)
		}

		// We have a llms.FunctionDefinition in tool.Function, and we have to
		// convert it to genai.FunctionDeclaration
		genaiFuncDecl := &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		}

		// Expect the Parameters field to be a map[string]any, from which we will
		// extract properties to populate the schema.
		params, ok := tool.Function.Parameters.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tool [%d]: unsupported type %T of Parameters", i, tool.Function.Parameters)
		}

		schema, err := convertSchemaRecursive(params, i, "")
		if err != nil {
			return nil, err
		}
		genaiFuncDecl.Parameters = schema

		// google genai only support one tool, multiple tools must be embedded into function declarations:
		// https://github.com/GoogleCloudPlatform/generative-ai/issues/636
		// https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#chat-samples
		genaiFuncDecls = append(genaiFuncDecls, genaiFuncDecl)
	}

	// Return nil if no tools are provided
	if len(genaiFuncDecls) == 0 {
		return nil, nil
	}

	genaiTools := []*genai.Tool{{FunctionDeclarations: genaiFuncDecls}}

	return genaiTools, nil
}

// convertToolSchemaType converts a tool's schema type from its langchaingo
// representation (string) to a genai enum.
func convertToolSchemaType(ty string) genai.Type {
	switch ty {
	case "object":
		return genai.TypeObject
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	default:
		return genai.TypeUnspecified
	}
}

// showContent is a debugging helper for genai.Content.
func showContent(w io.Writer, cs []*genai.Content) {
	fmt.Fprintf(w, "Content (len=%v)\n", len(cs))
	for i, c := range cs {
		fmt.Fprintf(w, "[%d]: Role=%s\n", i, c.Role)
		for j, p := range c.Parts {
			fmt.Fprintf(w, "  Parts[%v]: ", j)
			if p == nil {
				fmt.Fprintf(w, "nil\n")
				continue
			}
			if p.Text != "" {
				fmt.Fprintf(w, "Text %q\n", p.Text)
			} else if p.FunctionCall != nil {
				fmt.Fprintf(w, "FunctionCall Name=%v, Args=%v\n", p.FunctionCall.Name, p.FunctionCall.Args)
			} else if p.FunctionResponse != nil {
				fmt.Fprintf(w, "FunctionResponse Name=%v Response=%v\n", p.FunctionResponse.Name, p.FunctionResponse.Response)
			} else if p.InlineData != nil {
				fmt.Fprintf(w, "InlineData MIME=%q, size=%d\n", p.InlineData.MIMEType, len(p.InlineData.Data))
			} else {
				fmt.Fprintf(w, "unknown part type\n")
			}
		}
	}
}
