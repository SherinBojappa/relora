"""
The decoder model with a classification layer on top
Use last token in order to do classification
Since it does classification on the last token, it requires to know the position of the last token. If a
:obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
the last value in each row of the batch).
code based on https://github.com/huggingface/transformers/blob/285c6262a84490270d2f1a1c06ee9ccfc1b60e8f/src/transformers/models/gpt2/modeling_gpt2.py
"""
class DecoderForSequenceClassification(nn.Module):
    def __init__(self, num_labels, pretrained_model, n_emd, pad_token_id):
        self.transformer = pretrained_model
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.score = nn.Linear(n_emd, num_labels, bias=False)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        transformer_outputs = self.transformer(input_ids, attention_mask)
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        # find out where the last token is
        sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1

        pooled_logits = logits[range(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.view(-1), labels.to(self.dtype).view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        return loss, pooled_logits