import torch
import copy
import fairseq.models
import torch.nn as nn

# from fairseq.models.lstm import lstm_wiseman_iwslt_de_en, LSTMModel


class BaseActor(torch.nn.Module):
    def __init__(self, args, lan_size):
        super(BaseActor, self).__init__()
        self.args = args
        self.lan_size = lan_size
        # init
        self.bias = torch.nn.Linear(lan_size, 1)
        for p in self.bias.parameters():
            p.data.fill_(0.)

    def forward(self, feature, just_score=False):
        # input feature are 1 vector
        # feature: [1, lan_size]
        logits = self.bias.weight * feature
        output_logits = logits.view(1, -1)
        return output_logits


class PPLFeatureActor(torch.nn.Module):
    def __init__(self, args):
        super(PPLFeatureActor, self).__init__()
        self.args = args
        # init
        self.linear1 = torch.nn.Linear(2, 100)
        self.linear2 = torch.nn.Linear(100, 1)

    def forward(self, sample):
        # input feature are 1 vector
        src_feature = sample['gpt2_ppl']['src']
        tgt_feature = sample['gpt2_ppl']['tgt']
        a = src_feature.shape[0]
        new_src_feature = src_feature.view(a, 1)
        new_tgt_feature = tgt_feature[0].view(a, 1)
        act_fn = nn.Tanh()
        total_feature = torch.cat((new_src_feature, new_tgt_feature), 1)
        tot_logits = self.linear2(act_fn(self.linear1(total_feature)))

        for i in range(1, len(tgt_feature)):
            new_src_feature = src_feature.view(a, 1)
            new_tgt_feature = tgt_feature[i].view(a, 1)
            act_fn = nn.Tanh()
            total_feature = torch.cat((new_src_feature, new_tgt_feature), 1)
            sub_logits = self.linear2(act_fn(self.linear1(total_feature)))
            tot_logits = torch.cat((tot_logits, sub_logits), 1)
        return tot_logits


class EmbeddingActor1(torch.nn.Module):
    def __init__(self, lan_size):
        super(EmbeddingActor1, self).__init__()
        self.embedding = nn.Embedding(2, 1)
        torch.nn.init.zeros_(self.embedding.weight)

    def forward(self, feature, just_score=False):
        input = torch.tensor([0, 1]).cuda()
        return self.embedding(input).view(1, -1)


class EmbeddingActor2(torch.nn.Module):
    def __init__(self, lan_size):
        super(EmbeddingActor2, self).__init__()
        self.embedding = nn.Embedding(lan_size, 1)
        torch.nn.init.zeros_(self.embedding.weight)

    def forward(self, feature, just_score=False):
        input = feature.cuda()
        # input = torch.tensor([0, 1, 2]).cuda()
        return self.embedding(input).view(1, -1)


class LanguageActor(torch.nn.Module):
    def __init__(self, args, lan_size, optimize_emb=True):
        super(LanguageActor, self).__init__()
        self.args = args
        self.lan_size = lan_size
        embed_dim = args.data_actor_embed_dim
        if self.args.lan_embed_dim is None:
            lan_embed_dim = embed_dim
        else:
            lan_embed_dim = self.args.lan_embed_dim
        if self.args.embedding_file:
            self.lan_emb, fixed_embed_dim, embed_num = FixedEmbedding(self.args.embedding_file)
            assert embed_num == self.lan_size
            self.rand_lan_emb = Embedding(self.lan_size, lan_embed_dim - fixed_embed_dim)
            if not optimize_emb:
                self.lan_emb.weight.requires_grad = False
                self.rand_lan_emb.weight.requires_grad = False
        else:
            self.lan_emb = Embedding(self.lan_size, lan_embed_dim, None)
            if not optimize_emb:
                self.lan_emb.weight.requires_grad = False
        # init
        self.w = torch.nn.Linear(lan_embed_dim, embed_dim)
        self.project_out = torch.nn.Linear(embed_dim, 1)
        for p in self.w.parameters():
            torch.nn.init.uniform_(p, -0.1, 0.1)
        for p in self.project_out.parameters():
            torch.nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, feature):
        # input feature is lan id
        # feature: [1, 1]
        if self.args.embedding_file:
            emb_1 = self.lan_emb(feature)
            emb_2 = self.rand_lan_emb(feature)
            emb = torch.cat([emb_1, emb_2], dim=-1)
        else:
            emb = self.lan_emb(feature)
        x = self.w(emb)
        logits = self.project_out(x).squeeze(2)
        return logits


class AveEmbActor(torch.nn.Module):
    """Average Embedding actor"""

    def __init__(self, args, task, emb=None, optimize_emb=True):
        super(AveEmbActor, self).__init__()
        # assert task.source_dictionary == task.target_dictionary
        src_dictionary = task.source_dictionary
        trg_dictionary = task.target_dictionary
        self.padding_idx = src_dictionary.pad()
        if emb is None:
            embed_dim = args.data_actor_embed_dim
            if src_dictionary == trg_dictionary:
                num_embeddings = len(src_dictionary)
                self.src_embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
                self.trg_embed_tokens = self.src_embed_tokens
            else:
                num_embeddings = len(src_dictionary)
                self.src_embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
                num_embeddings = len(trg_dictionary)
                self.trg_embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.src_embed_tokens = emb
            self.trg_embed_tokens = emb
            embed_dim = emb.weight.size(1)
        self.project_out = torch.nn.Linear(2 * embed_dim, 1)
        self.out_score_type = args.out_score_type

    def forward(self, sample):
        src_tokens, trg_tokens = sample['net_input']['src_tokens'], sample['target']
        bsz, seqlen = src_tokens.size()

        src_word_count = (~src_tokens.eq(self.padding_idx)).long().sum(dim=-1, keepdim=True)
        # embed tokens
        x = self.src_embed_tokens(src_tokens)

        # B x T x C -> B x C
        x = x.sum(dim=1) / src_word_count.float()

        trg_word_count = (~trg_tokens.eq(self.padding_idx)).long().sum(dim=-1, keepdim=True)
        # embed tokens
        y = self.trg_embed_tokens(trg_tokens)

        # B x T x C -> B x C
        y = y.sum(dim=1) / trg_word_count.float()

        inp = torch.cat([x, y], dim=-1)
        # inp = y
        # B x 1
        if self.out_score_type == 'sigmoid':
            score = torch.sigmoid(self.project_out(inp))
        elif self.out_score_type == 'exp':
            score = torch.exp(self.project_out(inp))
        return score


def FixedEmbedding(embedding_file):
    embedding_data = []
    with open(embedding_file, 'r') as myfile:
        for line in myfile:
            toks = line.split()
            embedding_data.append([float(t) for t in toks])
    embedding_data = torch.FloatTensor(embedding_data)
    embedding_dim = embedding_data.size(1)
    num_embeddings = embedding_data.size(0)
    m = torch.nn.Embedding(num_embeddings, embedding_dim)
    m.weight = torch.nn.Parameter(embedding_data)
    return m, embedding_dim, num_embeddings


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
    if padding_idx is not None:
        torch.nn.init.constant_(m.weight[padding_idx], 0)
    return m


class LSTMActor(torch.nn.Module):
    """LSTM based actor"""

    def __init__(self, args, task):
        super(LSTMActor, self).__init__()
        # adapt args to LSTM model
        # args = lstm_wiseman_iwslt_de_en(args)
        # self.model = LSTMModel.build_model(args, task)

        self.model = fairseq.models.build_model(args, task)
        self.project_out = torch.nn.Linear(2 * args.data_actor_embed_dim, 1)
        self.out_score_type = args.out_score_type

    def forward(self, sample):
        net_output = self.model.encoder(**sample['net_input'])
        # B x 1
        inp = net_output[:, -1, :]
        if self.out_score_type == 'sigmoid':
            score = torch.sigmoid(self.project_out(inp))
        elif self.out_score_type == 'exp':
            score = torch.exp(self.project_out(inp))
        return score



