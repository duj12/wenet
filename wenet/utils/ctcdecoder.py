from typing import List, Tuple
from collections import defaultdict
import torch
from wenet.utils.common import log_add

class CTCDecoder:
    def __init__(self, beam_width: int =10, top_paths: int =10):
        self.beam_width = beam_width
        self.top_paths = top_paths

    def _ctc_prefix_beam_search(
        self,
        ctc_probs: torch.Tensor,
        ctc_lengths: torch.Tensor,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            ctc_probs:log_softmax output of ctc
            ctc_lengths: the seq lengths of this batch
        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        maxlen = ctc_probs.size(1)
        hyps = []
        max_hyp_len = []
        # iterate on batch dimension
        for ctc_prob in ctc_probs:
            # (maxlen, vocab_size)
            # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
            cur_hyps = [(tuple(), (0.0, -float('inf')))]
            # 2. CTC beam search step by step
            for t in range(0, maxlen):
                logp = ctc_prob[t]  # (vocab_size,)
                # key: prefix, value (pb, pnb), default value(-inf, -inf)
                next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
                # 2.1 First beam prune: select topk best
                top_k_logp, top_k_index = logp.topk(self.beam_width)  # (beam_size,)
                for s in top_k_index:
                    s = s.item()
                    ps = logp[s].item()
                    for prefix, (pb, pnb) in cur_hyps:
                        last = prefix[-1] if len(prefix) > 0 else None
                        if s == 0:  # blank
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pb = log_add([n_pb, pb + ps, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                        elif s == last:
                            #  Update *ss -> *s;
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pnb = log_add([n_pnb, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                            # Update *s-s -> *ss, - is for blank
                            n_prefix = prefix + (s, )
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)
                        else:
                            n_prefix = prefix + (s, )
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)

                # 2.2 Second beam prune
                next_hyps = sorted(next_hyps.items(),
                                   key=lambda x: log_add(list(x[1])),
                                   reverse=True)
                cur_hyps = next_hyps[:self.beam_width]
            hyp = []
            max_len = 0
            for y in cur_hyps:
                hyp.append((y[0], log_add([y[1][0], y[1][1]])))
                max_len = max(max_len, len(y[0]))
            hyps.append(hyp)
            max_hyp_len.append(max_len)
        return hyps, max_hyp_len


    def decode(self,
               inputs: torch.Tensor,
               sequence_length: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """ Decode the input data
        Args:
            inputs: shape [batch_size, max_step, num_classes], data_type torch.float32
            sequence_length: shape [batch_size]
        Returns:
           nbest_decoded: shape [batch_size, top_path, max_seq_len]
           log_probability: shape [batch_size, top_paths]
        """
        decoder_result, max_result_len = self._ctc_prefix_beam_search(inputs, sequence_length )
        n_nbest = []
        n_nlog_probs = []
        max_len = max(max_result_len)
        for i, result in  enumerate(decoder_result):
            nbest = []
            logprob = []
            for onebest in result:
                cand, logp = torch.tensor(onebest[0]), torch.tensor(onebest[1])
                leng = len(cand)
                pad = torch.nn.ConstantPad1d((0, max_len-leng), -1)
                nbest.append(pad(cand))
                logprob.append(logp)

            n_nbest.append(torch.stack(nbest, dim=0))
            n_nlog_probs.append(torch.tensor(logprob))

        return torch.stack(n_nbest, dim=0), torch.stack(n_nlog_probs, dim=0)

