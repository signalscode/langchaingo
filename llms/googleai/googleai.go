//nolint:all
package googleai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/tmc/langchaingo/internal/imageutil"
	"github.com/tmc/langchaingo/llms"
	"google.golang.org/genai"
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

	// Build the GenerateContentConfig
	config := g.buildGenerateContentConfig(&opts)

	// Convert messages to genai.Content
	contents, systemInstruction, err := g.convertMessages(messages)
	if err != nil {
		return nil, err
	}

	// Set system instruction if present
	if systemInstruction != nil {
		config.SystemInstruction = systemInstruction
	}

	var response *llms.ContentResponse

	if opts.StreamingFunc == nil {
		// Non-streaming request
		resp, err := g.client.Models.GenerateContent(ctx, opts.Model, contents, config)
		if err != nil {
			return nil, err
		}

		if len(resp.Candidates) == 0 {
			return nil, ErrNoContentInResponse
		}
		response, err = convertCandidates(resp.Candidates, resp.UsageMetadata)
		if err != nil {
			return nil, err
		}
	} else {
		// Streaming request
		response, err = g.generateContentStream(ctx, opts.Model, contents, config, &opts)
		if err != nil {
			return nil, err
		}
	}

	if g.CallbacksHandler != nil {
		g.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, response)
	}

	return response, nil
}

// buildGenerateContentConfig builds a GenerateContentConfig from CallOptions.
func (g *GoogleAI) buildGenerateContentConfig(opts *llms.CallOptions) *genai.GenerateContentConfig {
	temperature := float32(opts.Temperature)
	topP := float32(opts.TopP)
	topK := float32(opts.TopK)

	config := &genai.GenerateContentConfig{
		Temperature:     &temperature,
		TopP:            &topP,
		TopK:            &topK,
		CandidateCount:  int32(opts.CandidateCount),
		MaxOutputTokens: int32(opts.MaxTokens),
		StopSequences:   opts.StopWords,
	}

	// Configure safety settings
	config.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: g.opts.HarmThreshold.toGenAIHarmBlockThreshold(),
		},
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: g.opts.HarmThreshold.toGenAIHarmBlockThreshold(),
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: g.opts.HarmThreshold.toGenAIHarmBlockThreshold(),
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: g.opts.HarmThreshold.toGenAIHarmBlockThreshold(),
		},
	}

	// Configure tools if present
	if len(opts.Tools) > 0 {
		tools, err := convertTools(opts.Tools)
		if err == nil {
			config.Tools = tools
		}
	}

	// Set response MIME type
	switch {
	case opts.ResponseMIMEType != "" && opts.JSONMode:
		// Conflicting options - JSONMode takes precedence
		config.ResponseMIMEType = ResponseMIMETypeJson
	case opts.ResponseMIMEType != "":
		config.ResponseMIMEType = opts.ResponseMIMEType
	case opts.JSONMode:
		config.ResponseMIMEType = ResponseMIMETypeJson
	}

	// Support for cached content (if provided through metadata)
	if opts.Metadata != nil {
		if cachedContentName, ok := opts.Metadata["CachedContentName"].(string); ok && cachedContentName != "" {
			config.CachedContent = cachedContentName
		}
	}

	return config
}

// convertMessages converts langchaingo messages to genai.Content.
func (g *GoogleAI) convertMessages(messages []llms.MessageContent) ([]*genai.Content, *genai.Content, error) {
	var contents []*genai.Content
	var systemInstruction *genai.Content

	for _, mc := range messages {
		content, err := convertContent(mc)
		if err != nil {
			return nil, nil, err
		}

		// Handle system messages separately
		if mc.Role == llms.ChatMessageTypeSystem {
			systemInstruction = content
			continue
		}

		contents = append(contents, content)
	}

	return contents, systemInstruction, nil
}

// generateContentStream handles streaming content generation.
func (g *GoogleAI) generateContentStream(
	ctx context.Context,
	model string,
	contents []*genai.Content,
	config *genai.GenerateContentConfig,
	opts *llms.CallOptions,
) (*llms.ContentResponse, error) {
	var allParts []*genai.Part
	var lastCandidate *genai.Candidate
	var lastUsageMetadata *genai.GenerateContentResponseUsageMetadata

	for resp, err := range g.client.Models.GenerateContentStream(ctx, model, contents, config) {
		if err != nil {
			return nil, fmt.Errorf("error in stream mode: %w", err)
		}

		if len(resp.Candidates) == 0 {
			continue
		}

		candidate := resp.Candidates[0]
		lastCandidate = candidate
		lastUsageMetadata = resp.UsageMetadata

		if candidate.Content == nil {
			continue
		}

		for _, part := range candidate.Content.Parts {
			allParts = append(allParts, part)

			// Stream text content
			if part.Text != "" && !part.Thought {
				if opts.StreamingFunc(ctx, []byte(part.Text)) != nil {
					break
				}
			}
		}
	}

	if lastCandidate == nil {
		return nil, ErrNoContentInResponse
	}

	// Create a synthetic candidate with all accumulated parts
	mergedCandidate := &genai.Candidate{
		Content: &genai.Content{
			Parts: allParts,
			Role:  RoleModel,
		},
		FinishReason:     lastCandidate.FinishReason,
		SafetyRatings:    lastCandidate.SafetyRatings,
		CitationMetadata: lastCandidate.CitationMetadata,
	}

	return convertCandidates([]*genai.Candidate{mergedCandidate}, lastUsageMetadata)
}

// convertCandidates converts a sequence of genai.Candidate to a response.
func convertCandidates(candidates []*genai.Candidate, usage *genai.GenerateContentResponseUsageMetadata) (*llms.ContentResponse, error) {
	var contentResponse llms.ContentResponse

	for _, candidate := range candidates {
		var toolCalls []llms.ToolCall
		var thoughtParts []llms.ThoughtContent
		buf := strings.Builder{}

		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				// Handle thought parts (for Gemini 3+ models)
				if part.Thought {
					thoughtContent := llms.ThoughtContent{
						Text:      part.Text,
						Signature: part.ThoughtSignature,
					}
					thoughtParts = append(thoughtParts, thoughtContent)
					continue
				}

				// Handle text content
				if part.Text != "" {
					_, err := buf.WriteString(part.Text)
					if err != nil {
						return nil, err
					}
				}

				// Handle function calls
				if part.FunctionCall != nil {
					b, err := json.Marshal(part.FunctionCall.Args)
					if err != nil {
						return nil, err
					}
					toolCall := llms.ToolCall{
						FunctionCall: &llms.FunctionCall{
							Name:      part.FunctionCall.Name,
							Arguments: string(b),
						},
					}
					toolCalls = append(toolCalls, toolCall)
				}
			}
		}

		metadata := make(map[string]any)
		metadata[CITATIONS] = candidate.CitationMetadata
		metadata[SAFETY] = candidate.SafetyRatings

		if usage != nil {
			// Token count fields are int32 (not pointers)
			metadata["input_tokens"] = usage.PromptTokenCount
			metadata["output_tokens"] = usage.CandidatesTokenCount
			metadata["total_tokens"] = usage.TotalTokenCount
			// Standardized field names for cross-provider compatibility
			metadata["PromptTokens"] = usage.PromptTokenCount
			metadata["CompletionTokens"] = usage.CandidatesTokenCount
			metadata["TotalTokens"] = usage.TotalTokenCount

			// Cache-related token information (if available)
			if usage.CachedContentTokenCount > 0 {
				metadata["CachedTokens"] = usage.CachedContentTokenCount
				metadata["CacheReadInputTokens"] = usage.CachedContentTokenCount
				metadata["NonCachedInputTokens"] = usage.PromptTokenCount - usage.CachedContentTokenCount
			}

			// Thinking/reasoning token information
			if usage.ThoughtsTokenCount > 0 {
				metadata["ThinkingTokens"] = usage.ThoughtsTokenCount
			}
		}

		// Set reasoning content if thought parts exist
		var reasoningContent string
		if len(thoughtParts) > 0 {
			var reasoningBuilder strings.Builder
			for _, tp := range thoughtParts {
				if tp.Text != "" {
					reasoningBuilder.WriteString(tp.Text)
				}
			}
			reasoningContent = reasoningBuilder.String()
		}

		contentResponse.Choices = append(contentResponse.Choices,
			&llms.ContentChoice{
				Content:          buf.String(),
				StopReason:       string(candidate.FinishReason),
				GenerationInfo:   metadata,
				ToolCalls:        toolCalls,
				ReasoningContent: reasoningContent,
				ThoughtParts:     thoughtParts,
			})
	}
	return &contentResponse, nil
}

// convertParts converts between a sequence of langchain parts and genai parts.
func convertParts(parts []llms.ContentPart) ([]*genai.Part, error) {
	convertedParts := make([]*genai.Part, 0, len(parts))
	for _, part := range parts {
		var out *genai.Part

		switch p := part.(type) {
		case llms.TextContent:
			out = genai.NewPartFromText(p.Text)
		case llms.BinaryContent:
			out = genai.NewPartFromBytes(p.Data, p.MIMEType)
		case llms.ImageURLContent:
			typ, data, err := imageutil.DownloadImageData(p.URL)
			if err != nil {
				return nil, err
			}
			out = genai.NewPartFromBytes(data, typ)
		case llms.ToolCall:
			fc := p.FunctionCall
			var argsMap map[string]any
			if err := json.Unmarshal([]byte(fc.Arguments), &argsMap); err != nil {
				return convertedParts, err
			}
			out = genai.NewPartFromFunctionCall(fc.Name, argsMap)
		case llms.ToolCallResponse:
			out = genai.NewPartFromFunctionResponse(p.Name, map[string]any{
				"response": p.Content,
			})
		case llms.ThoughtContent:
			// Include thought content in subsequent requests
			out = &genai.Part{
				Text:             p.Text,
				Thought:          true,
				ThoughtSignature: p.Signature,
			}
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
		Parts: parts,
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

// convertTools converts from a list of langchaingo tools to a list of genai tools.
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
			if p.Text != "" {
				if p.Thought {
					fmt.Fprintf(w, "Thought %q\n", p.Text)
				} else {
					fmt.Fprintf(w, "Text %q\n", p.Text)
				}
			} else if p.InlineData != nil {
				fmt.Fprintf(w, "InlineData MIME=%q, size=%d\n", p.InlineData.MIMEType, len(p.InlineData.Data))
			} else if p.FunctionCall != nil {
				fmt.Fprintf(w, "FunctionCall Name=%v, Args=%v\n", p.FunctionCall.Name, p.FunctionCall.Args)
			} else if p.FunctionResponse != nil {
				fmt.Fprintf(w, "FunctionResponse Name=%v Response=%v\n", p.FunctionResponse.Name, p.FunctionResponse.Response)
			} else {
				fmt.Fprintf(w, "unknown part type\n")
			}
		}
	}
}
