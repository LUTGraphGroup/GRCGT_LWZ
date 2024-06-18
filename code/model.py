from torch_geometric.nn import GCN2Conv
from Rgcgt.layers import *


class TransformerModel(nn.Module):
    def __init__(
            self,
            hops,
            output_dim,
            input_dim,
            pe_dim,
            num_dis,
            num_meta,
            graphformer_layers,
            num_heads,
            hidden_dim,
            ffn_dim,
            dropout_rate,
            GCNII_layers

    ):

        super().__init__()
        self.seq_len = hops + 1
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        self.graphformer_layers = graphformer_layers

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.num_dis = num_dis
        self.num_meta = num_meta
        self.GCNII_layers = GCNII_layers
        self.convs = nn.ModuleList()
        for i in range(self.GCNII_layers):
            conv = GCN2Conv(channels=int(self.hidden_dim/2), alpha=0.1, theta=1, layer=i + 1)
            self.convs.append(conv)

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.num_heads)
            for _ in range(self.graphformer_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))

        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)

        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.output_dim)

        self.scaling = nn.Parameter(torch.ones(1) * 0.5)

        self.mlp = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 64),
        )
        self.decoder = InnerProductDecoder(self.output_dim, self.dropout_rate, self.num_dis)
        self.apply(lambda module: init_params(module, n_layers=self.graphformer_layers))

    def forward(self, processed_features, dis_data, meta_data):
        x_0_dis = dis_data.x
        x_dis = x_0_dis
        for conv in self.convs:
            x_dis = conv(x_dis, x_0_dis, dis_data.edge_index)

        x_0_meta = meta_data.x
        x_meta = x_0_meta
        for conv in self.convs:
            x_meta = conv(x_meta, x_0_meta, meta_data.edge_index)
        x_GCNII = torch.cat((x_dis, x_meta), dim=0)

        tensor = self.att_embeddings_nope(processed_features)  # Equation（18）
        # transformer encoder
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        x_former = self.final_ln(tensor)
        target = x_former[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(x_former, [1, self.seq_len - 1], dim=1)
        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        x_former = (node_tensor + neighbor_tensor).squeeze()

        output = torch.cat((x_GCNII, x_former), dim=1)
        embedings = self.mlp(output)
        x1 = self.decoder(embedings)
        return x1





