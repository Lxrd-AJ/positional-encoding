%% Attention heatmap

text = "The dog chased another dog";
[~, tokenizer] = bert();
[tokens, ~] = encode(tokenizer, text);

% create an embedding table