import torch
from wenet.utils.ctcdecoder import CTCDecoder
import editdistance

def edit_distance(hypothesis: torch.Tensor,
                  h_seq_lens: torch.Tensor,
                  truth: torch.Tensor,
                  t_seq_lens: torch.Tensor)-> torch.Tensor:
    # TODO: check tensor shape
    assert hypothesis.size(0) == truth.size(0)

    bs = hypothesis.size(0)
    hyp_max_seq_len = hypothesis.size(1)
    truth_max_seq_len = truth.size(1)

    assert h_seq_lens.size(0) == bs
    assert t_seq_lens.size(0) == bs

    distances = []

    for i, (hyp, ref) in enumerate(zip(hypothesis, truth)):
        hyp = hyp[:h_seq_lens[i]]
        distances.append(editdistance.eval(hyp, ref))
    return torch.tensor(distances, dtype=torch.int64)

class CTCMWERLoss():
    """ Computes the MWER (minimum WER) Loss.
        Reference: MINIMUM WORD ERROR RATE TRAINING FOR ATTENTION-BASED SEQUENCE-TO-SEQUENCE MODELS
        Rohit Prabhavalkar Tara N. Sainath Yonghui Wu Patrick Nguyen Zhifeng Chen Chung-Cheng Chiu Anjuli Kannan
        https://arxiv.org/pdf/1712.01818.pdf
    """

    def __init__(self, beam_width=8):
        """
        Args:
          beam_width: An int scalar >= 0 (beam search beam width).
        """
        self.beam_width = beam_width
        self.top_paths = beam_width
        self.ctc_prefix_beam_decoer = CTCDecoder(beam_width=beam_width,top_paths=beam_width)


    def loss(self, nbest_decoded, labels, labels_length, nbest_log_pdf):

        def word_error_number(nbest_decoded, labels, labels_length):
            # notes: labels shape [bs, max_seq_len] labels_length [bs]
            #        nbest_decoded, LIST[LIST[torch.Tensor]]
            # 1 convert  nbest decoded to shape [bs, max_seq_len], seq_len
            # 2 call editdistance

            w_e_n = []
            for (i,n) in enumerate(nbest_decoded):
                hyp_lens = torch.sum(torch.where(n==-1, 0, 1), dim=1)
                labels_repeat = labels[i].unsqueeze(0).repeat(n.size(0), 1)
                labels_length_repeat = labels_length[i].unsqueeze(0).repeat(n.size(0))
                w_e_n.append(edit_distance(n, hyp_lens, labels_repeat, labels_length_repeat).unsqueeze(0))

            return torch.cat(w_e_n, dim=0).to(torch.float32)

        # Computes log distribution.
        # log(sum(exp(elements across dimensions of a tensor)))
        sum_nbest_log_pdf = torch.logsumexp(nbest_log_pdf, 1, True) # (batch_size)
        # Re-normalized over just the N-best hypotheses.
        normal_nbest_pdf = torch.exp(nbest_log_pdf-sum_nbest_log_pdf) # [bs, top_path]

        # Number of word errors, but it represents by float.
        nbest_wen = word_error_number(
            nbest_decoded=nbest_decoded,
            labels=labels,
            labels_length=labels_length,
        ) # [bs, top_path]

        # Average number of word errors over the N-best hypohtheses
        mean_wen = torch.mean(nbest_wen, 1, True)

        # Re-normalized error word number over just the N-best hypotheses
        # normal_nbest_wen = [nbest_wen[k] -
        #                     mean_wen for k in range(self.top_paths)]
        normal_nbest_wen = nbest_wen-mean_wen # [bs, top_path]

        # Expected number of word errors over the training set.

        mwer_loss = normal_nbest_pdf * normal_nbest_wen
        return torch.sum(mwer_loss)

    def forward(self, logits, labels, logit_length, label_length):
        """
        Args:
          logits: tensor of shape [batch_size, max_seq_len, vocal_size].
          labels: tensor of shape [batch_size, max_label_len]
          logit_length: tensor of shape [batch_size] Length of input sequence in
            logits.
          label_length: tensor of shape [batch_size] Length of labels sequence.
        Returns:
          loss: tensor, MWER loss.
        """

        # Beam search for top-N hypotheses.
        #   decoded: A list of length top_paths.
        #   log_probabilities: A float matrix [batch_size, top_paths]
        #                     containing sequence log-probabilities.
        nbest_decoded, log_probabilities = self.ctc_prefix_beam_decoer.decode(
            logits, logit_length)
        nbest_log_pdf = log_probabilities

        return self.loss(nbest_decoded, labels, label_length, nbest_log_pdf)
