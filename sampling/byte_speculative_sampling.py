import torch
from tqdm import tqdm
import torch

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder, ByteDecoder

@torch.no_grad()
def byte_speculative_sampling(prefix : torch.Tensor, byte_prefix: torch.Tensor,
                              approx_model : torch.nn.Module, target_model : torch.nn.Module,
                              max_len : int , gamma : int = 4,
                              temperature : float = 1, top_k : int = 0, top_p : float = 0, 
                              verbose : bool = True , random_seed : int = None) -> torch.Tensor:
    """
    huggingface version Speculative Sampling. (Assisted generation)
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    assert prefix.shape[0] == 1, "input batch size must be 1"
    assert byte_prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)

    byte_draft_count = 0
    accepted_count = 0
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]
        byte_prefix_len = byte_prefix.shape[1]
        x = approx_model_cache.generate(byte_prefix, gamma)
        

        byte_draft_count += (x.shape[1] -  byte_prefix.shape[1]) # maybe need to consider the maximun length of the output tokens
        
        # convert the byte prefix to token prefix
        byte_drafted_x = x[:, byte_prefix_len:]
        # sentence piece tokenizer is anoyying...
        draft_str = ByteDecoder().decode(byte_drafted_x)
        if draft_str[0] == " ": draft_str = draft_str[1:]
        draft_x = torch.cat([prefix, Decoder().encode(draft_str, return_tensors='pt').to(device)], dim=1)
        # check at which point draft_x and prefix are different
        
            
        _ = target_model_cache.generate(draft_x, 1)
        draft_token_len = draft_x.shape[1] - prefix_len

        # torch.multinomial(target_model_cache._prob_history[0,:,:], num_samples=1).squeeze(1)[None, :]
        # sample the token from the last token of prefix
        selected_tokens = torch.multinomial(target_model_cache._prob_history[0,-draft_token_len-1:,:], num_samples=1).squeeze(1)[None, :] # sample(target_model_cache._prob_history)https://github.com/ChaosCodes/LLMSpeculativeSampling.git
        n_matches = ((~(draft_x[:, -draft_token_len:] == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
        # Limit the generated tokens number to T
        if prefix_len + n_matches + 1 >= T:
            n_matches = T - prefix_len - 1
        
        # byte accepted prefix
        byte_prefix = ByteDecoder().encode(Decoder().decode(draft_x[:, :prefix_len + n_matches]), return_tensors='pt').to(device)
        print("--")
        print(n_matches)
        print(Decoder().decode(prefix))
        print(Decoder().decode(draft_x))
        # byte resampled prefix
        resample_byte_token = ByteDecoder().encode(Decoder().decode(torch.cat([prefix, selected_tokens], dim=1)[:,  :prefix_len + n_matches + 1]), return_tensors='pt').to(device)

        print(ByteDecoder().decode(resample_byte_token))
        resample_byte_token = resample_byte_token[:, byte_prefix.shape[1]:]
        approx_model_cache.rollback(byte_prefix.size()[1])
        target_model_cache.rollback(prefix_len + n_matches)
    
        
        accepted_byte_prefix_len = byte_prefix.shape[1] - byte_prefix_len
        accepted_count += accepted_byte_prefix_len

        # concat thee resample byte token
        byte_prefix = torch.cat((byte_prefix, resample_byte_token), dim=1)
        prefix = torch.cat((draft_x[:, :prefix_len + n_matches], selected_tokens[:, n_matches].unsqueeze(0)), dim=1)
        # heuristic adjust gamma
        # if accepted_byte_prefix_len >= gamma - 3:
        #     gamma += 8
        # else:
        #     gamma = max(5, gamma - 1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, total_drafted_tokens {byte_draft_count}")
    return prefix

@torch.no_grad()
def byte_speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    break
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)

    return prefix

