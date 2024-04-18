import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from ete3 import Tree

# DNA states
STATES = ["A", "G", "C", "T"]
STATE_TO_IDX = {"A": 0, "G": 1, "C": 2, "T": 3}
n_states = len(STATES)

# neural network parameters
learning_rate = 0.01

# traverse stack parameters
d_out_traverse = 4
d_hidden_traverse = 32

# site-aggregate transformer parameters
nhead = 2
d_model = 2 * d_out_traverse  # size of embedding that we feed into transformer, i.e.
# 2 * (length of mutation vector), from concatenating in two directions across edge
dim_feedforward = 8
layer_count = 4


class TraverseNN(L.LightningModule):
    """
    A pytorch module which takes a list of ete3.Trees as input and outputs a list of
    vectors of 0's and 1's to indicate whether each edge of each input tree is maximally
    parsimonious or not, respectively, for the sequences assigned to the leaf nodes.

    The forward function applies two traversals to the input tree, first root-ward and
    then leaf-ward. The traversals combine data across the tree for each site, keeping
    data from separate sites separate. Then, a transformer encoder is used to combine
    data across separate sites, at each node. The transformer encoder outputs (n_sites)-
    many vectors but we only keep the data in the first vector. Finally, this encoder
    output vector is passed through a linear classifier and a sigmoid to get a
    prediction.

    Attributes:
        traverse_stack: MLP with single hidden layer, used to summarize mutation data
            below a given node at a given site, by combining data from its two children
        encoder_layer:
        encoder: transformer encoder used to summarize mutation data across all sizes,
            at a given node
        classifier: linear layer to produce prediction logit, for whether each edge is
            present in a maximum parsimony tree
    """

    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate
        self.traverse_stack = nn.Sequential(
            nn.Linear(2 * n_states + 2 * d_out_traverse, d_hidden_traverse),
            nn.ReLU(),
            nn.Linear(d_hidden_traverse, d_out_traverse),
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.classifier = nn.Linear(d_model, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        xb, yb = train_batch
        pred = torch.stack([self.forward_on_tree(item) for item in xb])
        loss = F.binary_cross_entropy_with_logits(pred, yb.unsqueeze(-1))
        self.log("train_loss", loss, batch_size=len(xb), on_epoch=True)
        # log predictions on positive- and negative-datapoints, and show data in console
        # progress bar
        prob_predictions = F.sigmoid(pred)
        pos_predictions = prob_predictions[yb < 0.5]
        neg_predictions = prob_predictions[yb >= 0.5]
        self.log("pos_prediction_avg", torch.mean(pos_predictions), prog_bar=True)
        self.log("neg_prediction_avg", torch.mean(neg_predictions), prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        xb, yb = val_batch
        pred = torch.stack([self.forward_on_tree(item) for item in xb])
        loss = F.binary_cross_entropy_with_logits(pred, yb.unsqueeze(-1))
        self.log("val_loss", loss, batch_size=len(xb))

    def forward(self, input, optimized=False):
        """
        Takes an iterable of ete3.Tree as input and outputs a tensor of 0's and 1's to
        indicate whether each input tree is maximally parsimonious or not, respectively,
        for the sequences assigned to the leaf nodes.
        Args:
            input (list of Trees): has attribute to_parent["edge_mutation"] on each node,
                which is a torch tensor that encodes the mutation between the node and
                its parent, e.g. A -> G is encoded by [-1, 1, 0, 0]
            optimized (boolean): if True, runs more efficiently by skipping type check
                that allows ete.Tree input
        """
        if not optimized:
            if type(input) == Tree:
                logit = self.forward_on_tree(input)
                return F.sigmoid(logit)
        # assume input is a list (or iterable) of trees
        logits = torch.cat([self.forward_on_tree(item) for item in input])
        return F.sigmoid(logits)

    def forward_on_tree(self, tree: Tree):
        """
        Takes an ete3.Tree as input and outputs a list of 0's and 1's to indicate
        whether each edge on the input tree is present in a maximally parsimonious tree
        or not, respectively, for the sequences assigned to the leaf nodes.
        Args:
            tree (ete3 Tree): each node has a sequence attribute, which is a string
                consisting of the characters A, G, C, T
        """
        self.assign_mutation_vectors(tree)
        self.compute_features_via_traversal(tree, len(tree.sequence))
        encoder_output = self.site_aggregate(tree)
        # encoder_output dim = (n_nodes, 1, 8)
        logit = self.classifier(encoder_output[:, 0])
        return logit

    def compute_features_via_traversal(
        self,
        tree: Tree,
        seq_length,
        feature_name="edge_mutation",
    ):
        """
        Takes an ete3.Tree as input and assigns a feature to each node which is a tensor
        of dimension (n_edges, n_sites, n_states=4). At each edge and each site, the
        tensor encodes a summary of the mutations that occurred on the subtree on either
        side of the specified edge, at the specified site.
        Args:
            tree (ete3 Tree): each node has a torch tensor attribute
                to_parent["edge_mutation"] that encodes the mutation between the node and
                its parent, e.g. A -> G is encoded by [-1, 1, 0, 0]
            seq_length (int): length of input sequences
            feature_name (string): name of feature assigned to nodes of the tree that
                are used for encoding
        """
        # root-ward traversal
        for node in tree.traverse(strategy="postorder"):
            if node.is_leaf():
                node.to_parent["clade_mutation_feature"] = torch.zeros(
                    (seq_length, d_out_traverse)
                )
            elif len(node.children) == 1:  # node is root with single child
                assert node.up is None
                child = node.children[0]
                node.to_parent["clade_mutation_feature"] = child.to_parent[
                    "clade_mutation_feature"
                ]
            else:
                feature = torch.zeros((seq_length, d_out_traverse))
                try:
                    child1, child2 = node.children
                except ValueError:
                    raise ValueError(
                        f"Input tree must be bifurcating, but node has"
                        "{len(node.children)} children"
                    )
                for i in range(seq_length):
                    feature[i] = self.node_aggregate(
                        child1.to_parent, child2.to_parent, feature_name, site_idx=i
                    )
                node.to_parent["clade_mutation_feature"] = feature
        # leaf-ward traversal
        for node in tree.traverse(strategy="preorder"):
            feature = torch.zeros((seq_length, d_out_traverse))
            if node.is_root():
                node.from_parent["clade_mutation_feature"] = feature
            elif node.up.is_root():
                assert (
                    len(node.up.children) == 1
                ), "Error: root of tree should have single child"
                node.from_parent["clade_mutation_feature"] = feature
            else:
                parent = node.up
                # `node` should have a single sister node
                assert len(node.get_sisters()) == 1
                sister = node.get_sisters()[0]
                for i in range(seq_length):
                    feature[i] = self.node_aggregate(
                        parent.from_parent, sister.to_parent, feature_name, site_idx=i
                    )
                node.from_parent["clade_mutation_feature"] = feature
        return tree

    def site_aggregate(self, tree):
        """
        Takes a tensor encoding site-wise mutations on subclades of a tree and
        aggregates its n_sites using a Transformer
        """
        input_features = torch.stack(
            [
                torch.cat(
                    (
                        node.to_parent["clade_mutation_feature"],
                        node.from_parent["clade_mutation_feature"],
                    ),
                    dim=1,
                )
                for node in tree.traverse(strategy="preorder")
            ]
        )
        # input_features dim = (n_nodes, n_sites, d_model=8)
        ## debug
        # print("input:", input_features)
        # print("input dim:", input_features.size())
        out = self.encoder(input_features)
        # out dim = (n_nodes, n_sites, d_model=8)
        return out

    def node_aggregate(
        self,
        first_node_dict,
        second_node_dict,
        feature_name,
        site_idx,
        symmetrize=False,
    ):
        """
        Takes in dictionaries of feature vectors from two neighbor-nodes of a given
        node, and outputs the `clade_mutation_feature` vector for that node.
        The two neighbor-nodes can be either:
            - two children of a node, during root-ward traversal, or
            - one parent and one sister of a node, during leaf-ward traversal.
        """
        i = site_idx
        first_data = torch.cat(
            (
                first_node_dict[feature_name][i],
                first_node_dict["clade_mutation_feature"][i],
            ),
            dim=0,
        )
        second_data = torch.cat(
            (
                second_node_dict[feature_name][i],
                second_node_dict["clade_mutation_feature"][i],
            ),
            dim=0,
        )
        combined_data = torch.cat((first_data, second_data))
        output = self.traverse_stack(combined_data)
        if symmetrize:
            output += self.traverse_stack(torch.cat((first_data, second_data)))
        return output

    @staticmethod
    def assign_mutation_vectors(tree):
        """
        Modifies input tree by adding a `to_parent` and `from_parent` dict attributes.
        `to_parent["edge_mutation"]` is a size (n_sites, n_states=4) torch.tensor which
        records the mutation from the node's parent to the (child) node,
        e.g., a mutation `A -> T` is encoded as [...,[-1, 0, 0, 1],...]
        Args:
            tree (ete3 Tree): has sequence attribute on each node
        Returns: None
        """
        n_sites = len(tree.sequence)
        for node in tree.traverse():
            for i in range(n_sites):
                mut_vec = [0.0, 0.0, 0.0, 0.0]
                if node.up is None:  # node is root
                    pass
                else:  # non-root node
                    n_seq = node.sequence[i]
                    p_seq = node.up.sequence[i]
                    # except AttributeError:
                    #     n_seq = node.name
                    #     p_seq = node.up.name
                    try:
                        mut_vec[STATE_TO_IDX[n_seq]] += 1
                        mut_vec[STATE_TO_IDX[p_seq]] -= 1
                    except KeyError:
                        raise ValueError(f"Each node sequence must be in {STATES}")
                new_row = torch.tensor(mut_vec).unsqueeze(0)
                if i == 0:
                    node.add_feature("to_parent", {"edge_mutation": new_row})
                    node.add_feature("from_parent", {"edge_mutation": -new_row})
                else:
                    node.to_parent["edge_mutation"] = torch.cat(
                        (node.to_parent["edge_mutation"], new_row)
                    )
                    node.from_parent["edge_mutation"] = torch.cat(
                        (node.from_parent["edge_mutation"], -new_row)
                    )
        return None


class TransformerEncoderTraversal(TraverseNN):
    """
    A pytorch module which takes a list of ete3.Trees as input and outputs 0's and 1's
    to indicate whether each input tree is maximally parsimonious or not, respectively,
    for the sequences assigned to the leaf nodes.

    The forward function first encodes the mutation features using a transformer encoder
    and then applies two traversals to the input tree, first root-ward and then leaf-ward

    Attributes:
        encoder
        classifier
        traverse_stack
    """

    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate=learning_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_states,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.classifier = nn.Linear(d_model, 1)

    def forward_on_tree(self, tree: Tree):
        """
        Takes an ete3.Tree as input and outputs a 0 or 1 to indicate whether the input
        tree is maximally parsimonious or not, respectively, for the sequences assigned
        to the leaf nodes.
        Args:
            tree (ete3 Tree): each node has a torch tensor attribute
                to_parent["edge_mutation"] that encodes the mutation between the node and
                its parent, e.g. A -> G is encoded by [-1, 1, 0, 0]
        """
        self.assign_mutation_vectors(tree)
        self.site_aggregate(tree)
        self.compute_features_via_traversal(
            tree, seq_length=1, feature_name="all_sites_edge_mutation"
        )
        output = torch.stack(
            [
                torch.cat(
                    (
                        node.to_parent["clade_mutation_feature"],
                        node.from_parent["clade_mutation_feature"],
                    ),
                    dim=1,
                ).squeeze()
                for node in tree.traverse(strategy="preorder")
            ]
        )
        # output dim = (n_nodes, d_model=8)
        logit = self.classifier(output)
        return logit

    def site_aggregate(self, tree: Tree):
        for node in tree.traverse(strategy="preorder"):
            input = node.to_parent["edge_mutation"]
            # input dim = (n_sites, n_states=4)
            output = self.encoder(input)
            # output dim = (n_sites, n_states=4)
            # assign learned features to nodes
            node.to_parent["all_sites_edge_mutation"] = output
            node.from_parent["all_sites_edge_mutation"] = -output
        return None
