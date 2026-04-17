package watsonx

import (
	"context"
	"errors"

	wx "github.com/IBM/watsonx-go/pkg/models"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
)

var (
	ErrInvalidPrompt = errors.New("invalid prompt")
	ErrEmptyResponse = errors.New("no response")
)

type LLM struct {
	CallbacksHandler callbacks.Handler
	client           *wx.Client

	modelID string
}

var _ llms.Model = (*LLM)(nil)

// Call implements the LLM interface.
func (wx *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, wx, prompt, options...)
}

// GenerateContent implements the Model interface.
func (wx *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) { //nolint: lll, cyclop, whitespace

	cb := callbacks.EffectiveHandler(ctx, wx.CallbacksHandler)
	if cb != nil {
		cb.HandleLLMGenerateContentStart(ctx, messages)
	}

	prompt, err := getPrompt(messages)
	if err != nil {
		if cb != nil {
			cb.HandleLLMError(ctx, err)
		}
		return nil, err
	}

	result, err := wx.client.GenerateText(
		wx.modelID,
		prompt,
		toWatsonxOptions(&options)...,
	)
	if err != nil {
		if cb != nil {
			cb.HandleLLMError(ctx, err)
		}
		return nil, err
	}

	if result.Text == "" {
		if cb != nil {
			cb.HandleLLMError(ctx, ErrEmptyResponse)
		}
		return nil, ErrEmptyResponse
	}

	resp := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				Content: result.Text,
			},
		},
	}
	if cb != nil {
		cb.HandleLLMGenerateContentEnd(ctx, resp)
	}
	return resp, nil
}

func New(modelID string, opts ...wx.ClientOption) (*LLM, error) {
	c, err := wx.NewClient(opts...)
	if err != nil {
		return nil, err
	}

	return &LLM{
		client:  c,
		modelID: modelID,
	}, nil
}

func getPrompt(messages []llms.MessageContent) (string, error) {
	// Assume we get a single text message
	msg0 := messages[0]
	part := msg0.Parts[0]
	prompt, ok := part.(llms.TextContent)
	if !ok {
		return "", ErrInvalidPrompt
	}

	return prompt.Text, nil
}

func getDefaultCallOptions() *llms.CallOptions {
	return &llms.CallOptions{
		TopP:              -1,
		TopK:              -1,
		Temperature:       -1,
		Seed:              -1,
		RepetitionPenalty: -1,
		MaxTokens:         -1,
	}
}

func toWatsonxOptions(options *[]llms.CallOption) []wx.GenerateOption {
	opts := getDefaultCallOptions()
	for _, opt := range *options {
		opt(opts)
	}

	o := []wx.GenerateOption{}
	if opts.TopP != -1 {
		o = append(o, wx.WithTopP(opts.TopP))
	}
	if opts.TopK != -1 {
		o = append(o, wx.WithTopK(uint(opts.TopK)))
	}
	if opts.Temperature != -1 {
		o = append(o, wx.WithTemperature(opts.Temperature))
	}
	if opts.Seed != -1 {
		o = append(o, wx.WithRandomSeed(uint(opts.Seed)))
	}
	if opts.RepetitionPenalty != -1 {
		o = append(o, wx.WithRepetitionPenalty(opts.RepetitionPenalty))
	}
	if opts.MaxTokens != -1 {
		o = append(o, wx.WithMaxNewTokens(uint(opts.MaxTokens)))
	}
	if len(opts.StopWords) > 0 {
		o = append(o, wx.WithStopSequences(opts.StopWords))
	}

	/*
	   watsonx options not supported:

	   	wx.WithMinNewTokens(minNewTokens)
	   	wx.WithDecodingMethod(decodingMethod)
	   	wx.WithLengthPenalty(decayFactor, startIndex)
	   	wx.WithTimeLimit(timeLimit)
	   	wx.WithTruncateInputTokens(truncateInputTokens)
	   	wx.WithReturnOptions(inputText, generatedTokens, inputTokens, tokenLogProbs, tokenRanks, topNTokens)
	*/

	return o
}
