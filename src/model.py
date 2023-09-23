#!/usr/bin/env python3 

import torch
from typing import Dict
import warnings


class Model(torch.nn.Module):
    """
    Create Model object from embedder, decoder,
    and unembedder (if not None).

    Args
    ----
    embedder: src.embedder.make_embedder
        Instance of embedder class.
    decoder: src.decoder.make_decoder
        Instance of decoder class.
    unembedder: src.unembedder.make_unembedder
        Instance of unembedder class.
        Only added to model if not None.

    Methods
    ----
    forward(batch: Dict[str, torch.tensor])
        Forward pass of model.
    prep_batch(batch: Dict[str, torch.tensor])
        Prepare batch for forward pass.
    compute_loss(batch: Dict[str, torch.tensor])
        Compute training loss.
    from_pretrained(pretrained_path: str)
        Load pretrained model from pretrained_path.
        Needs to point to pytorch_model.bin file 
    """
    def __init__(
        self,
        embedder: torch.nn.Module,
        decoder: torch.nn.Module,
        unembedder: torch.nn.Module = None
        ) -> torch.nn.Module:
        
        super().__init__()
        self.name = f'Embedder-{embedder.name}_Decoder-{decoder.name}'
        self.embedder = embedder
        self.decoder = decoder
        self.unembedder = unembedder
        self.is_decoding_mode = False

    def from_pretrained(
        self,
        pretrained_path: str
        ) -> None:
        """Load pretrained model from pretrained_path.
        Needs to point to pytorch_model.bin file.
        """
        print(
            f'Loading pretrained model from {pretrained_path}'
        )

        if next(self.parameters()).is_cuda:
            pretrained = torch.load(pretrained_path)

        else:
            pretrained = torch.load(pretrained_path, map_location=torch.device('cpu'))
        
        for k in self.state_dict():
            
            if k in pretrained:
                assert pretrained[k].shape == self.state_dict()[k].shape,\
                    f'{k} shape mismatch between pretrained model and current model '+\
                    f'{pretrained[k].shape} vs {self.state_dict()[k].shape}'
        
        for k in pretrained:
            
            if k not in self.state_dict():
                warnings.warn(
                    f'Warning: /!\ Skipping {k} from {pretrained_path} '\
                    'because it is not part of the current model'
                )

        # we set strict=False, because we can be sure
        # that all relevant keys are in pretrained
        self.load_state_dict(pretrained, strict=False)

    def switch_decoding_mode(
        self,
        is_decoding_mode: bool = False,
        num_decoding_classes: int = None
        ) -> None:
        """Switch model to decoding model or back to training mode.
        Necessary to adapt pre-trained models to downstream
        decoding tasks.
        
        Args
        ----
        is_decoding_mode: bool
            Whether to switch to decoding mode or not.
        num_decoding_classes: int
            Number of classes to use for decoding.    
        """
        self.is_decoding_mode = is_decoding_mode
        self.embedder.switch_decoding_mode(is_decoding_mode=is_decoding_mode)
        self.decoder.switch_decoding_mode(
            is_decoding_mode=is_decoding_mode,
            num_decoding_classes=num_decoding_classes
        )

    def compute_loss(
        self,
        batch: Dict[str, torch.tensor],
        return_outputs: bool = False
        ) -> Dict[str, torch.tensor]:
        """
        Compute training loss, based on 
        embedder's training-style.

        Args
        ----
        batch: Dict[str, torch.tensor]
            Input batch (as generated by src.batcher)
        return_outputs: bool
            Whether to return outputs of forward pass
            or not. If False, only loss is returned.

        Returns
        ----
        losses: Dict[str, torch.tensor]
            Training losses.
        outputs: torch.tensor
            Outputs of forward pass.
        """
        (outputs, batch) = self.forward(
            batch=batch,
            return_batch=True
        )
        losses = self.embedder.loss(
            batch=batch,
            outputs=outputs
        )

        return (losses, outputs) if return_outputs else losses

    def prep_batch(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        """Prepare input batch for forward pass.
        Calls src.embedder.prep_batch.
        
        Args
        ----
        batch: Dict[str, torch.tensor]
            Input batch (as generated by src.batcher)
        """
        return self.embedder.prep_batch(batch=dict(batch))

    def forward(
        self,
        batch: Dict[str, torch.tensor],
        prep_batch: bool = True,
        return_batch: bool = False
        ) -> torch.tensor:
        """
        Forward pass of model.
        
        Args
        ----
        batch: Dict[str, torch.tensor]
            Input batch (as generated by src.batcher)
        prep_batch: bool
            Whether to prep batch for forward pass
            by calling self.embedder.prep_batch
        return_batch: bool
            Whether to return batch after forward pass
            or not. If False, only outputs of forward pass
            are returned.

        Returns
        ----
        outputs: torch.tensor
            Outputs of forward pass.
        batch: Dict[str, torch.tensor]
            Input batch (as returned by prep_batch, 
            if prep_batch is True)
        """
        
        if prep_batch:
            batch = self.prep_batch(batch=batch)
        
        else:
            assert 'inputs_embeds' in batch, 'inputs_embeds not in batch'

        data = batch["inputs_embeds"]

        batch['inputs_embeds'] = self.embedder(batch=batch)
        outputs = self.decoder(batch=batch)
        
        if self.unembedder is not None and not self.is_decoding_mode:
            outputs['outputs'] = self.unembedder(inputs=outputs['outputs'])['outputs']
        
        return (outputs, batch) if return_batch else outputs
