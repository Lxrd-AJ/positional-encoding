%% Create tokenizer
[~, tokenizer] = bert();
iTokenizer = struct(tokenizer);

%% create the token embeddings
vocabSize = iTokenizer.VocabularyEncoding.NumWords;
text = "The dog chased another dog";

[tokens, ~] = encode(tokenizer, text);
disp([tokenizer.StartToken, tokenizer.StartCode])
disp(tokens)

embedDims = 64;
embeddingLayer = dlnetwork([...
    sequenceInputLayer(1, "MinLength", 1)
    wordEmbeddingLayer(embedDims, vocabSize) ...
]);

% Prepare the tokens for the forward pass
tokens = dlarray(tokens{:}, 'CT');
tokenEmbeddings = forward(embeddingLayer, tokens);

%% Crude positional encoding
numTokens = size(tokenEmbeddings, 2);
positions = (1:numTokens) ./ numTokens;
posTokenEmbeddings = tokenEmbeddings + positions;

