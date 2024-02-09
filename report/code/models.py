class GRU4RecExtAbstractEntitiesCLFSUM(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GRU4RecExtAbstractEntitiesCLFSUM, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        #self.selected_features = config['selected_features']
        #self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        #self.num_feature_field = len(config['selected_features'])

        self.loss_type = config['loss_type']

        self.score_calc = config["score_calc"]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        pretrained_news_emb = dataset.get_preload_weight('nid')
        pretrained_abs_emb = dataset.get_preload_weight('aid')
        pretrained_clf_emb = dataset.get_preload_weight('cid')
        pretrained_sum_emb = dataset.get_preload_weight('sid')

        self.abstract_embed_layer = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_abs_emb)).to(dtype=torch.float)
        self.abstract_embed_size = self.abstract_embed_layer.embedding_dim

        self.feature_embed_layer = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_news_emb).to(dtype=torch.float))
        self.feature_embed_size = self.feature_embed_layer.embedding_dim

        self.clf_embed_layer = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_clf_emb)).to(dtype=torch.float)
        self.clf_embed_size = self.clf_embed_layer.embedding_dim

        self.sum_embed_layer = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_sum_emb)).to(dtype=torch.float)
        self.sum_embed_size = self.sum_embed_layer.embedding_dim

        self.item_gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        # For simplicity, we use same architecture for item_gru and feature_gru
        self.feature_gru_layers = nn.GRU(
            input_size=self.feature_embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )


        self.abstract_gru_layers = nn.GRU(
            input_size=self.abstract_embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.clf_gru_layers = nn.GRU(
            input_size=self.clf_embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.sum_gru_layers = nn.GRU(
            input_size=self.sum_embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.dense_layer = nn.Linear(self.hidden_size * 5, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['feature_embed_layer', 'abstract_embed_layer']

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.dropout(item_seq_emb)
        item_gru_output, _ = self.item_gru_layers(item_seq_emb_dropout)  # [B Len H]

        feature_seq_emb = self.feature_embed_layer(item_seq)
        feature_gru_output, _ = self.feature_gru_layers(feature_seq_emb)  # [B Len H]

        abstract_seq_emb = self.abstract_embed_layer(item_seq)
        abstract_gru_output, _ = self.abstract_gru_layers(abstract_seq_emb)

        clf_seq_emb = self.clf_embed_layer(item_seq)
        clf_gru_output, _ = self.clf_gru_layers(clf_seq_emb)

        sum_seq_emb = self.sum_embed_layer(item_seq)
        sum_gru_output, _ = self.sum_gru_layers(sum_seq_emb)

        output_concat = torch.cat((item_gru_output, 
                                   feature_gru_output, 
                                   abstract_gru_output, 
                                   clf_gru_output,
                                   sum_gru_output), -1)  # [B Len 3*H]
        output = self.dense_layer(output_concat)
        output = self.gather_indexes(output, item_seq_len - 1)  # [B H]
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)  # [B H]
            neg_items_emb = self.item_embedding(neg_items)  # [B H]
            if self.score_calc == "mean":
                pos_score = torch.mean(seq_output * pos_items_emb, dim=-1)  # [B]
                neg_score = torch.mean(seq_output * neg_items_emb, dim=-1)  # [B]
            else: # self.score_calc == "sum"
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    

def run_recbole(config_dict):
    config = Config(model='GRU4Rec', dataset='mind_small', config_dict=config_dict)

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)

    # write config info into log
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    if config_dict["additional_feat_suffix"] == ["newsemb", "absemb", "clfembb", "sumemb"]:
        print("Using abstract, title entities embeddings, embeddings from classification and embeddings of summarized texts")
        model = GRU4RecExtAbstractEntitiesCLFSUM(config, train_data.dataset).to(config['device'])
    elif config_dict["additional_feat_suffix"] == ["newsemb", "absemb", "clfembb"]:
        print("Using abstract, title entities embeddings and embeddings from classification")
        model = GRU4RecExtAbstractEntitiesCLF(config, train_data.dataset).to(config["device"])

    elif config_dict["additional_feat_suffix"] == ["newsemb", "absemb"]:
        print("Using abstract and title entities embeddings")
        model = GRU4RecExtAbstractEntities(config, train_data.dataset).to(config["device"])
    else: # ["newsemb"]
        print("Using only title entities embeddings")
        model = GRU4RecExtTitleEntities(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, 
                                                      valid_data=valid_data, 
                                                      show_progress=True)
    return best_valid_score, best_valid_result



if __name__ == '__main__':
    config_dict = load_config('configs/sequential/gru4rec_ext.yaml')
    run_recbole(config_dict=config_dict)
    update_markdown('GRU4Rec', 'mind_small', 'Exp. 5')
    