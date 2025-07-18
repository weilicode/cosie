import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphAutoencoder(nn.Module):

    """
    Graph autoencoder architecture for learning embeddings.

    This class implements a symmetric encoder-decoder structure using GNN layers (e.g., GCNConv).

    Parameters
    ----------
    encoder_dim : list of int
        A list specifying the hidden and output dimensions of each GNN layer in the encoder.
    
    activation : str, optional
        Activation function to use between GNN layers. Must be one of {'relu', 'sigmoid', 'tanh', 'leakyrelu'}.
        Default is `'relu'`.
    
    base_model : nn.Module, optional
        The GNN layer to use. Must follow the PyTorch Geometric GNN interface (e.g., `GCNConv`, `SAGEConv`, `GATConv`).
        Default is `GCNConv`.

    Methods
    -------
    encoder(x, edge_index)
        Encode input features into latent embedding.
    
    decoder(x, edge_index)
        Decode latent features back into original feature space.
    
    forward(x, edge_index)
        Perform full encoder-decoder reconstruction and return output.
    """

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 base_model = GCNConv):   
       
        super(GraphAutoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._base_model = base_model
        self.encoder_conv = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()

        self.encoder_conv.append(base_model(encoder_dim[0], encoder_dim[1]))
        for i in range(1, self._dim):
            self.encoder_conv.append(base_model(encoder_dim[i], encoder_dim[i+1]))


        decoder_dim = [i for i in reversed(encoder_dim)]
        self.decoder_conv.append(base_model(decoder_dim[0], decoder_dim[1]))
        for i in range(1, self._dim):
            self.decoder_conv.append(base_model(decoder_dim[i], decoder_dim[i+1]))

        if activation == 'relu':
            self._activation = nn.ReLU()
        elif activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation == 'tanh':
            self._activation = nn.Tanh()
        elif self._activation == 'leakyrelu':
            self._activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def encoder(self, x, edge_index):
        """
        Encode input node features into latent embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input node feature matrix of shape (n_nodes, in_dim).
        
        edge_index : torch.Tensor
            Edge index defining the graph connectivity.

        Returns
        -------
        x : torch.Tensor
            Latent node embeddings of shape (n_nodes, latent_dim).
        """
        for i in range(self._dim):
            if i == self._dim - 1:
                x = self.encoder_conv[i](x, edge_index)
            else:
                x = self.encoder_conv[i](x, edge_index)
                x = self._activation(x)
        x = F.normalize(x, p=2, dim=1)   
        return x

    def decoder(self, x, edge_index):
        """
        Decode latent embeddings to reconstruct original node features.

        Parameters
        ----------
        x : torch.Tensor
            Latent node embeddings of shape (n_nodes, latent_dim).
        
        edge_index : torch.Tensor
            Edge index defining the graph connectivity.

        Returns
        -------
        x : torch.Tensor
            Reconstructed node features of shape (n_nodes, in_dim).
        """
        for i in range(self._dim):
            x = self.decoder_conv[i](x, edge_index)
            if i < self._dim - 1:
                x = self._activation(x)
        return x

    def forward(self, x, edge_index):
        """
        Perform full encoder-decoder forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input node feature matrix of shape (n_nodes, in_dim).
        
        edge_index : torch.Tensor
            Edge index defining the graph connectivity.

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed node feature matrix of shape (n_nodes, in_dim).
        """
        latent = self.encoder(x, edge_index)   
        x_hat = self.decoder(latent, edge_index)



class Prediction_mlp(nn.Module):

    """
    A fully connected multi-layer perceptron (MLP) for cross-modality prediction between latent embeddings.

    Parameters
    ----------
    prediction_dim : list of int
        A list defining the hidden dimensions of the MLP prediction module.
    
    activation : str, optional
        Activation function to use between hidden layers. Must be one of
        `{'relu', 'sigmoid', 'tanh', 'leakyrelu'}`. Default is `'relu'`.

    Methods
    -------
    forward(x)
        Apply the MLP to an input embedding and return predicted embedding.
    """

    def __init__(self,
                 prediction_dim,
                 activation='relu'):

        super(Prediction_mlp, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if i < self._depth - 1:
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)


    def forward(self, x):
        """
        Perform full forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input cell embedding.
        
        Returns
        -------
        output : torch.Tensor
            Predicted cell embedding.
        """
        output = self._encoder(x)
        output = F.normalize(output, p=2, dim=1)
        return output

