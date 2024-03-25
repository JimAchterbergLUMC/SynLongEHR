# replace discriminator and add masking layer in Gretel Synthetics torch_modules.py
# includes masking in the discriminator to allow variable sequence lengths
# DONT FORGET TO IMPLEMENT MASKING PARAMETERS IN THE DGAN.PY SCRIPT IN DOPPELGANGER PACKAGE

import torch


class Discriminator(torch.nn.Module):
    """Discriminator network for DGAN model."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 5,
        num_units: int = 200,
        features: bool = False,
        max_seq_len: int = 40,
        total_attributes: int = 12,
    ):
        """Create discriminator network.

        Args:
            input_dim: size of input to discriminator network
            num_layers: # of layers in MLP used for discriminator
            num_units: # of units per layer in MLP used for discriminator
            features: whether features are discriminated
            max_seq_len: maximum length of sequences (relevant iff features=True)
            total_attributes: # attributes + # additional attributes
        """
        super(Discriminator, self).__init__()

        self.features = features  # denotes whether to use masking layer
        self.max_seq_len = max_seq_len  # needed for mask computation
        self.total_attributes = total_attributes  # needed for mask computation

        seq = []
        # mask discriminator input if we are dealing with features
        if self.features:
            seq.append(
                MaskingLayer(
                    mask_value=1,
                    max_seq_len=self.max_seq_len,
                    total_attributes=self.total_attributes,
                )
            )

        last_dim = input_dim
        for _ in range(num_layers):
            seq.append(torch.nn.Linear(int(last_dim), int(num_units)))
            seq.append(torch.nn.ReLU())
            last_dim = num_units

        seq.append(torch.nn.Linear(int(last_dim), 1))

        self.seq = torch.nn.Sequential(*seq)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply module to input.

        Args:
            input: input tensor of shape (batch size, input_dim)

        Returns:
            Discriminator output with shape (batch size, 1).
        """

        return self.seq(input)


class MaskingLayer(torch.nn.Module):
    """Masking layer to model mixed length sequences"""

    def __init__(
        self,
        mask_value: int = 1,
        discriminator: bool = True,
        max_seq_len: int = 40,
        total_attributes: int = 12,
    ):
        super().__init__()
        self.mask_value = mask_value
        self.max_seq_len = max_seq_len
        self.total_attributes = total_attributes

    def forward(self, input):

        # Applies the mask after the last column corresponds to the mask value for the first time.

        # we are only applying the masking layer if we are discriminating features
        # in case of discriminator the input dimension is (B,a+aa+f*t)
        # so we first need to ensure we are applying the mask to 3d features

        feat = input[:, self.total_attributes :]
        feat = torch.reshape(
            feat,
            (input.shape[0], self.max_seq_len, int(feat.shape[1] / self.max_seq_len)),
        )
        # features are now 3d in the discriminator as well, to apply the mask

        # mask is all ones except AFTER the gen flag occurs, then it is all zeros
        # do this by cumsumming the gen flag and 'inverting' it (zeros followed by ones -> ones followed by zeros)
        mask = (torch.cumsum(feat[:, :, -1], dim=1) != self.mask_value).float()
        # but we dont want to mask the flag step itself
        flag_cols = torch.argmin(mask, dim=1)
        rows = torch.arange(mask.shape[0])
        mask[rows, flag_cols] = self.mask_value

        # expand the mask to span all columns of the features
        mask = mask.unsqueeze(-1).expand(feat.shape)

        # apply the mask to the features
        feat = mask * feat

        # we need to turn output back into correct format (add attributes)
        feat = torch.reshape(feat, input[:, self.total_attributes :].shape)
        output = torch.cat([input[:, : self.total_attributes], feat], dim=-1)

        return output
