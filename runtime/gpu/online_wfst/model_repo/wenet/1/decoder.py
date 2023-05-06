import os, multiprocessing
import torch
from typing import List
from riva.asrlib.decoder.python_decoder import (BatchedMappedDecoderCuda,
                                                BatchedMappedOnlineDecoderCuda,
                                                BatchedMappedDecoderCudaConfig)
from frame_reducer import FrameReducer
def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.
    See description of make_non_pad_mask.
    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.
    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def remove_duplicates_and_blank(hyp: List[int],
                                eos: int,
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id and hyp[cur] != eos:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp

def ctc_greedy_search(ctc_probs, encoder_out_lens, vocabulary, blank_id, eos):
    batch_size, maxlen = ctc_probs.size()[:2]
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.cpu().masked_fill_(mask, eos)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    hyps = [remove_duplicates_and_blank(hyp, eos, blank_id) for hyp in hyps]
    total_hyps = []
    for hyp in hyps:
        total_hyps.append("".join([vocabulary[i] for i in hyp]))
    return total_hyps

def load_word_symbols(path):
    word_id_to_word_str = {}
    with open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            word_str, word_id = line.rstrip().split()
            word_id_to_word_str[int(word_id)] = word_str
    return word_id_to_word_str

def create_decoder_config():
    config = BatchedMappedDecoderCudaConfig()
    config.n_input_per_chunk = 50
    config.online_opts.decoder_opts.default_beam = 17.0
    config.online_opts.decoder_opts.lattice_beam = 8.0
    config.online_opts.decoder_opts.max_active = 10_000
    config.online_opts.determinize_lattice = True
    config.online_opts.max_batch_size = 200
    config.online_opts.num_channels = config.online_opts.max_batch_size * 2
    config.online_opts.frame_shift_seconds = 0.04
    config.online_opts.lattice_postprocessor_opts.acoustic_scale = 1.0
    config.online_opts.lattice_postprocessor_opts.lm_scale = 1.0
    config.online_opts.lattice_postprocessor_opts.word_ins_penalty = 0.0
    config.online_opts.lattice_postprocessor_opts.nbest = 1
    config.online_opts.num_decoder_copy_threads = 2
    config.online_opts.num_post_processing_worker_threads = (
        multiprocessing.cpu_count() - config.online_opts.num_decoder_copy_threads
    )

    return config

class RivaWFSTDecoder:
    def __init__(self, vocab_size, tlg_dir, config_dict, nbest=1):
        config = create_decoder_config()

        # config.online_opts.lattice_postprocessor_opts.nbest = nbest
        # config.online_opts.decoder_opts.lattice_beam = config_dict['lattice_beam']
        # config.online_opts.lattice_postprocessor_opts.acoustic_scale = config_dict['acoustic_scale'] # noqa
        # config.n_input_per_chunk = config_dict['n_input_per_chunk']
        # config.online_opts.decoder_opts.default_beam = config_dict['default_beam']
        # config.online_opts.decoder_opts.max_active = config_dict['max_active']
        # config.online_opts.determinize_lattice = config_dict['determinize_lattice']
        # config.online_opts.max_batch_size = config_dict['max_batch_size']
        # config.online_opts.num_channels = config_dict['num_channels']
        # config.online_opts.frame_shift_seconds = config_dict['frame_shift_seconds']
        # config.online_opts.lattice_postprocessor_opts.lm_scale = config_dict['lm_scale']
        # config.online_opts.lattice_postprocessor_opts.word_ins_penalty = config_dict['word_ins_penalty'] # noqa

        # config.online_opts.decoder_opts.blank_penalty = -5.0
        # offline_decoder = BatchedMappedDecoderCuda(
        #     config, "/ws/onnx_model/TLG.fst",
        #     "/ws/onnx_model/words.txt", 12000
        # )
        # TODO: load fail when using "pip install -e ." from source riva-asrlib-decoder code
        # load fail only happen in tritonserver, run outside is normal
        print(f"Create RivaWFSTDecoder...{tlg_dir}, {vocab_size}")
        self.decoder = BatchedMappedOnlineDecoderCuda(
            config.online_opts, os.path.join(tlg_dir, "TLG.fst"),
            os.path.join(tlg_dir, "words.txt"), vocab_size
        )
        print("RivaWFSTDecoder: \n", self.decoder)
        self.word_id_to_word_str = load_word_symbols(os.path.join(tlg_dir, "words.txt"))
        self.nbest = nbest
        self.vocab_size = vocab_size
        self.frame_reducer = FrameReducer(0.98)


    def decode(self, logits, length, is_first_chunk=True, is_last_chunk=True ):
        # TODO: prepare is_first_chunk and is_last_chunk
        logits, length = self.frame_reducer(logits, length.cuda(), logits)
        # logits[:,:,0] -= 2.0
        logits = logits.to(torch.float32).contiguous()
        cpu_lengths = length.to(torch.long).to('cpu').contiguous()

        batch_size = logits.shape[0]
        # batch_size = 1
        # TODO: Remove this!
        cpu_lengths = cpu_lengths[:batch_size]
        corr_ids = list(range(batch_size))
        for corr_id in corr_ids:
            success = self.decoder.try_init_corr_id(corr_id)
            # Do SetLatticeCallback() here if you want
            # Is there some way that I can get the lattice other than callbacks?
            assert success
        # Are my results going to be contiguous in general?
        # log_probs_ptrs = [0] * batch_size
        log_probs_list = [0] * batch_size
        is_first_chunk = [0] * batch_size
        is_last_chunk = [0] * batch_size
        # print("GALVEZ:")
        # torch.cuda.synchronize()
        for i in range(batch_size):
            # Do I need itemsize() here???
            # log_probs_ptrs[i] = log_probs.data_ptr() + log_probs.stride(0) * log_probs.element_size() * i
            log_probs_list[i] = logits[i, :cpu_lengths[i], :]
            is_first_chunk[i] = True
            is_last_chunk[i] = True

        channels, partial_hypotheses = \
                    self.decoder.decode_batch(corr_ids, log_probs_list,
                                         is_first_chunk, is_last_chunk)
        total_hyps = []
        for ph in partial_hypotheses:
            hyp = "".join(self.word_id_to_word_str[word]
                      for word in ph.words if word != 0)
            total_hyps.append(hyp)
        return total_hyps

if __name__ == "__main__":
    config = create_decoder_config()
    decoder = BatchedMappedOnlineDecoderCuda(
        config.online_opts, "/ws/onnx_model/TLG.fst",
        "/ws/onnx_model/words.txt", 12000
    )
    print(config)
    print(decoder)