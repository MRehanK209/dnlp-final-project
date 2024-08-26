import sys

import torch

sys.path.append("../")
from bert1 import BertModel

"""
This script tests the correct functionality of the BERT model by comparing the output of
a forward pass with a reference.

Can only be called from same directory as the file.

The test is successful if the output of the forward pass is close to the reference.
"""

sanity_data = torch.load("sanity_check.data")

# text_batch = ["hello world", "hello neural network for NLP"]
# Tokenizer here
sent_ids = torch.tensor(
    [[101, 7592, 2088, 102, 0, 0, 0, 0], [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]]
)
att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

# Load our model and run it
bert = BertModel.from_pretrained("bert-base-uncased")
outputs = bert(sent_ids, att_mask)
att_mask = att_mask.unsqueeze(-1)
outputs["last_hidden_state"] = outputs["last_hidden_state"] * att_mask
sanity_data["last_hidden_state"] = sanity_data["last_hidden_state"] * att_mask

# Compare outputs
for k in ["last_hidden_state", "pooler_output"]:
    assert torch.allclose(outputs[k], sanity_data[k], atol=1e-5, rtol=1e-3)
print("Your BERT implementation is correct!")
