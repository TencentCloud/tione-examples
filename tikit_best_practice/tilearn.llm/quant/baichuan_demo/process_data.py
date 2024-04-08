import torch
import torch.nn as nn
from datasets import load_dataset
import random

def get_wikitext2(nsamples, seed, seqlen, tokenizer, test_data):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    traindata = load_dataset(test_data, 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset(test_data, 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append({"input_ids": inp})
    return trainloader, testenc

def get_loaders(test_data, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """
    Select the appropriate loader based on dataset name.

    Args:
        name (str): The name of the dataset ('wikitext2', 'c4', or 'ptb').
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader and encoded validation/test set.
    """
    return get_wikitext2(nsamples, seed, seqlen, tokenizer, test_data)

def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    """
    Evaluate perplexity (ppl) specifically on the wikitext dataset.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        testenc (TokenizerWrapper): Encoded input IDs from test set.
        bs (int): Batch size for evaluation.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the wikitext test dataset.
    """
    # Get input IDs from the TokenizerWrapper instance
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()  # Example: [cat, sat, on, ???] -> [cat, sat, on]
        shift_labels = inputs[:, 1:]  # Example: [The, cat, sat, on] -> [cat, sat, on]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)  # nll = loss * sequence_length * batch_size

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(
        torch.stack(nlls).sum() / (nsamples * model.seqlen))  # ppl = exp(∑(nlls) / (num_samples * sequence_length))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

def eval_ppl(model, tokenizer, test_data, seed, device=torch.device("cuda:0")):
    """
    Evaluate perplexity (ppl) on a specified model and tokenizer.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the test dataset.
    """

    # Print status
    print(f"evaluating on {test_data}")

    # Get the test loader
    _, testloader = get_loaders(
        test_data, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer
    )

    # Evaluate perplexity in no grad context to avoid updating the model
    with torch.no_grad():
        # Perplexity measures how well the probability distribution predicted by the model aligns with the actual distribution of the words. Lower perplexity is better.
        ppl = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl