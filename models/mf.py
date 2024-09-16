from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import LitDataModule
from base_model import LitModel
from datasets import ML100K


class MatrixFactorization(nn.Module):
    def __init__(self, embedding_dims, num_users, num_items, 
                 sparse=False, **kwargs):
        super().__init__()
        self.sparse = sparse
        
        self.user_embedding = nn.Embedding(num_users, embedding_dims, sparse=sparse)
        self.user_bias = nn.Embedding(num_users, 1, sparse=sparse)
        
        self.item_embedding = nn.Embedding(num_items, embedding_dims, sparse=sparse)
        self.item_bias = nn.Embedding(num_items, 1, sparse=sparse) 

        for param in self.parameters():
            nn.init.normal_(param, std=0.01)   

    def forward(self, user_id, item_id):
        Q = self.user_embedding(user_id)
        bq = self.user_bias(user_id).flatten()

        I = self.item_embedding(item_id)
        bi = self.item_bias(item_id).flatten()

        return (Q*I).sum(-1) + bq + bi


class LitMF(LitModel):
    # super().__init__()
    def get_loss(self, pred_ratings, batch):
        return F.mse_loss(pred_ratings, batch[-1])

    def update_metric(self, m_outputs, batch):
        _, _, gt = batch
        self.rmse.update(m_outputs, gt)

    def forward(self, batch):
        user_ids, item_ids, _ = batch
        return self.model(user_ids, item_ids)
        


def main(args):
    # Setup data module
    data = LitDataModule(ML100K(), batch_size=args.batch_size)
    data.setup()

    # Setup model
    model = LitMF(MatrixFactorization, sparse=False, 
                  num_users=data.num_users, num_items=data.num_items,
                  embedding_dims=args.embedding_dims)
    
    # Logger setup
    logger = TensorBoardLogger("lightning_logs", name=f"MF_{args.embedding_dims}")

    # Create the trainer using manually parsed arguments
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
    
        precision=args.precision if args.precision else 32,  # Add precision if available
        # Add other Trainer arguments as necessary
    )

    # Train the model
    trainer.fit(model, data)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Add custom arguments
    parser.add_argument("--embedding_dims", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=10)  # Example trainer argument
   
    parser.add_argument("--precision", type=int, default=32, help="Precision (16 or 32)")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)