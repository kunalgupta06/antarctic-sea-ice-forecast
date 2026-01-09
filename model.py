"""
Deep Learning Architecture for Antarctic Sea-Ice Forecasting
Components:
1. CNN Encoder - Spatial feature extraction
2. ConvLSTM - Spatio-temporal dynamics
3. Temporal Transformer - Long-range dependencies
4. Multi-Task Decoder - Ice forecast + Habitat risk
5. Uncertainty Head - Epistemic uncertainty estimation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Skip connection if dimensions match
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class CNNEncoder(nn.Module):
    """
    CNN Encoder for spatial feature extraction from sea-ice images
    Uses a hierarchical structure to capture multi-scale patterns
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        channel_progression: Tuple[int, ...] = (32, 64, 128, 256),
        kernel_size: int = 3
    ):
        super().__init__()
        
        self.channels = channel_progression
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, channel_progression[0], 7, padding=3),
            nn.BatchNorm2d(channel_progression[0]),
            nn.ReLU(inplace=True),
        )
        
        # Encoder blocks with downsampling
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        for i in range(len(channel_progression) - 1):
            self.encoder_blocks.append(
                ConvBlock(channel_progression[i], channel_progression[i + 1], kernel_size)
            )
            self.downsample.append(nn.MaxPool2d(2, 2))
        
        # Final projection
        self.final_conv = nn.Conv2d(channel_progression[-1], channel_progression[-1], 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) input image
        Returns:
            features: (B, C_out, H', W') encoded features
            skip_connections: list of intermediate features for decoder
        """
        skip_connections = []
        
        x = self.initial(x)
        skip_connections.append(x)
        
        for block, down in zip(self.encoder_blocks, self.downsample):
            x = block(x)
            skip_connections.append(x)
            x = down(x)
        
        x = self.final_conv(x)
        
        return x, skip_connections


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell for spatio-temporal modeling
    Maintains spatial structure while learning temporal dynamics
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Combined gates for efficiency
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,  # i, f, o, g gates
            kernel_size,
            padding=padding,
            bias=bias
        )
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input
            hidden: tuple of (h, c) each (B, hidden_dim, H, W)
        Returns:
            h_next, c_next: next hidden and cell states
        """
        B, _, H, W = x.shape
        
        if hidden is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        else:
            h, c = hidden
        
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        # Split into gates
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM for learning climate temporal dynamics
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
        return_all_layers: bool = False
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.return_all_layers = return_all_layers
        
        layers = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            layers.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
        
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, T, C, H, W) input sequence
        Returns:
            output: (B, T, hidden_dim, H, W) or last timestep only
            hidden_states: list of final (h, c) for each layer
        """
        B, T, C, H, W = x.shape
        
        if hidden is None:
            hidden = [None] * self.num_layers
        
        outputs = []
        
        for t in range(T):
            input_t = x[:, t]
            
            for layer_idx, (layer, h) in enumerate(zip(self.layers, hidden)):
                input_t, c = layer(input_t, h)
                hidden[layer_idx] = (input_t, c)
                
                if layer_idx < self.num_layers - 1:
                    input_t = self.dropout(input_t)
            
            outputs.append(input_t)
        
        outputs = torch.stack(outputs, dim=1)  # (B, T, C, H, W)
        
        return outputs, hidden


class PositionalEncoding2D(nn.Module):
    """2D Positional encoding for spatial features"""
    
    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        
        pe = torch.zeros(d_model, max_h, max_w)
        
        # Height encoding
        h_pos = torch.arange(0, max_h).unsqueeze(1).float()
        w_pos = torch.arange(0, max_w).unsqueeze(0).float()
        
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        
        pe[0::4, :, :] = torch.sin(h_pos * div_term.unsqueeze(0).unsqueeze(-1)).permute(2, 0, 1)[:d_model // 4]
        pe[1::4, :, :] = torch.cos(h_pos * div_term.unsqueeze(0).unsqueeze(-1)).permute(2, 0, 1)[:d_model // 4]
        pe[2::4, :, :] = torch.sin(w_pos * div_term.unsqueeze(0).unsqueeze(0)).permute(2, 0, 1)[:d_model // 4]
        pe[3::4, :, :] = torch.cos(w_pos * div_term.unsqueeze(0).unsqueeze(0)).permute(2, 0, 1)[:d_model // 4]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        _, _, H, W = x.shape
        return x + self.pe[:, :H, :W].unsqueeze(0)


class TemporalTransformerBlock(nn.Module):
    """Transformer block for temporal attention across climate states"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) temporal sequence
            mask: optional attention mask
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class TemporalTransformer(nn.Module):
    """
    Transformer for capturing long-range temporal dependencies in climate data
    Operates on temporally-pooled spatial features
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 365
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Temporal position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) temporal features
        Returns:
            (B, T, D) transformed features
        """
        B, T, D = x.shape
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class SeaIceDecoder(nn.Module):
    """
    Decoder to reconstruct sea-ice concentration maps from encoded features
    Uses transposed convolutions with skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        channel_progression: Tuple[int, ...] = (256, 128, 64, 32),
        out_channels: int = 1
    ):
        super().__init__()
        
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i in range(len(channel_progression) - 1):
            self.decoder_blocks.append(
                ConvBlock(channel_progression[i] * 2, channel_progression[i + 1])  # *2 for skip connections
            )
            self.upsample.append(
                nn.ConvTranspose2d(channel_progression[i], channel_progression[i], 2, stride=2)
            )
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(channel_progression[-1], channel_progression[-1], 3, padding=1),
            nn.BatchNorm2d(channel_progression[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_progression[-1], out_channels, 1),
            nn.Sigmoid()  # Output in [0, 1] for ice concentration
        )
        
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) encoded features
            skip_connections: list of encoder features for skip connections
        Returns:
            (B, 1, H_out, W_out) sea-ice concentration prediction
        """
        for i, (block, up) in enumerate(zip(self.decoder_blocks, self.upsample)):
            x = up(x)
            
            if skip_connections is not None and i < len(skip_connections):
                skip = skip_connections[-(i + 2)]  # Reverse order
                # Handle size mismatch
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            else:
                x = torch.cat([x, x], dim=1)  # Duplicate if no skip
            
            x = block(x)
        
        return self.final(x)


class HabitatRiskHead(nn.Module):
    """
    Multi-task head for predicting penguin habitat risk
    Based on sea-ice features and ecological priors
    """
    
    def __init__(self, in_channels: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        self.spatial_pool = nn.AdaptiveAvgPool2d(8)
        
        self.risk_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 8 * 8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 5),  # 5 risk categories
            nn.Softmax(dim=-1)
        )
        
        # Also output a spatial risk map
        self.spatial_risk = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) encoded features
        Returns:
            risk_class: (B, 5) probability distribution over risk levels
            risk_map: (B, 1, H, W) spatial risk map
        """
        pooled = self.spatial_pool(x)
        risk_class = self.risk_network(pooled)
        risk_map = self.spatial_risk(x)
        
        return risk_class, risk_map


class UncertaintyHead(nn.Module):
    """
    Uncertainty estimation head using heteroscedastic approach
    Outputs mean and log-variance for each prediction
    """
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        
        self.mean_head = nn.Conv2d(in_channels, 1, 1)
        self.logvar_head = nn.Conv2d(in_channels, 1, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: (B, 1, H, W) predicted mean
            logvar: (B, 1, H, W) log-variance (uncertainty)
        """
        mean = torch.sigmoid(self.mean_head(x))
        logvar = self.logvar_head(x)
        
        return mean, logvar


class AntarcticSeaIceForecaster(nn.Module):
    """
    Complete model for Antarctic sea-ice forecasting
    
    Architecture:
    Input Sequence -> CNN Encoder -> ConvLSTM -> Transformer -> Decoders
                                                      |
                                                      ├-> Sea-Ice Forecast
                                                      ├-> Habitat Risk
                                                      └-> Uncertainty
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        in_channels: int = 1,
        cnn_channels: Tuple[int, ...] = (32, 64, 128, 256),
        convlstm_hidden: int = 128,
        convlstm_layers: int = 2,
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        dropout: float = 0.1,
        use_uncertainty: bool = True,
        prediction_horizon: int = 30
    ):
        super().__init__()
        
        self.image_size = image_size
        self.use_uncertainty = use_uncertainty
        self.prediction_horizon = prediction_horizon
        self.transformer_dim = transformer_dim
        
        # Spatial encoder
        self.encoder = CNNEncoder(in_channels, cnn_channels)
        
        # Calculate encoded spatial size
        self.encoded_h = image_size[0] // (2 ** (len(cnn_channels) - 1))
        self.encoded_w = image_size[1] // (2 ** (len(cnn_channels) - 1))
        
        # Temporal models
        self.convlstm = ConvLSTM(
            cnn_channels[-1], convlstm_hidden,
            kernel_size=3, num_layers=convlstm_layers, dropout=dropout
        )
        
        # Project ConvLSTM output to transformer dim
        self.temporal_proj = nn.Conv2d(convlstm_hidden, transformer_dim, 1)
        
        # Fixed pooled spatial size for transformer
        self.pooled_size = 4  # Pool to 4x4
        self.temporal_feature_dim = transformer_dim * self.pooled_size * self.pooled_size
        
        self.transformer = TemporalTransformer(
            d_model=self.temporal_feature_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Prediction head for future timesteps
        self.future_predictor = nn.Sequential(
            nn.Linear(self.temporal_feature_dim, transformer_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_dim * 4, cnn_channels[-1] * self.encoded_h * self.encoded_w)
        )
        
        # Decoders
        self.ice_decoder = SeaIceDecoder(cnn_channels[-1], cnn_channels[::-1])
        self.habitat_head = HabitatRiskHead(cnn_channels[-1])
        
        if use_uncertainty:
            self.uncertainty_head = UncertaintyHead(cnn_channels[-1])
        
    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of images"""
        B, T, C, H, W = x.shape
        
        # Encode each frame
        x = x.view(B * T, C, H, W)
        encoded, skip = self.encoder(x)
        
        # Reshape back
        _, C_enc, H_enc, W_enc = encoded.shape
        encoded = encoded.view(B, T, C_enc, H_enc, W_enc)
        
        return encoded, skip
    
    def forward(
        self,
        x: torch.Tensor,
        forecast_steps: int = None
    ) -> dict:
        """
        Args:
            x: (B, T, C, H, W) input sequence
            forecast_steps: number of future steps to predict
        Returns:
            dict with predictions
        """
        if forecast_steps is None:
            forecast_steps = self.prediction_horizon
            
        B, T, C, H, W = x.shape
        
        # Encode spatial features
        encoded, skip_connections = self.encode_sequence(x)
        
        # ConvLSTM for temporal dynamics
        lstm_out, hidden_states = self.convlstm(encoded)  # (B, T, C, H', W')
        
        # Project and prepare for transformer
        lstm_out_reshaped = lstm_out.view(B * T, -1, self.encoded_h, self.encoded_w)
        projected = self.temporal_proj(lstm_out_reshaped)
        
        # Pool spatial dims for transformer (fixed size)
        pooled = F.adaptive_avg_pool2d(projected, (self.pooled_size, self.pooled_size))
        pooled = pooled.view(B, T, -1)  # (B, T, transformer_dim * pooled_size^2)
        
        # Temporal transformer
        temporal_features = self.transformer(pooled)  # (B, T, D)
        
        # Predict future states
        predictions = {
            'ice_maps': [],
            'habitat_risk_class': [],
            'habitat_risk_map': [],
        }
        
        if self.use_uncertainty:
            predictions['uncertainty'] = []
        
        # Use last temporal feature to predict future
        last_feature = temporal_features[:, -1]  # (B, D)
        
        for step in range(forecast_steps):
            # Predict future spatial features
            future_features = self.future_predictor(last_feature)
            future_features = future_features.view(B, -1, self.encoded_h, self.encoded_w)
            
            # Decode to ice map
            ice_map = self.ice_decoder(future_features)
            predictions['ice_maps'].append(ice_map)
            
            # Habitat risk
            risk_class, risk_map = self.habitat_head(future_features)
            predictions['habitat_risk_class'].append(risk_class)
            predictions['habitat_risk_map'].append(risk_map)
            
            # Uncertainty
            if self.use_uncertainty:
                _, logvar = self.uncertainty_head(future_features)
                predictions['uncertainty'].append(torch.exp(0.5 * logvar))
        
        # Stack predictions
        predictions['ice_maps'] = torch.stack(predictions['ice_maps'], dim=1)
        predictions['habitat_risk_class'] = torch.stack(predictions['habitat_risk_class'], dim=1)
        predictions['habitat_risk_map'] = torch.stack(predictions['habitat_risk_map'], dim=1)
        
        if self.use_uncertainty:
            predictions['uncertainty'] = torch.stack(predictions['uncertainty'], dim=1)
        
        return predictions
    
    def predict_long_term(
        self,
        x: torch.Tensor,
        years: int = 50,
        samples_per_year: int = 12
    ) -> dict:
        """
        Long-term autoregressive forecasting
        
        Args:
            x: (B, T, C, H, W) initial sequence
            years: number of years to forecast
            samples_per_year: predictions per year (e.g., monthly)
        """
        total_steps = years * samples_per_year
        
        all_predictions = {
            'ice_maps': [],
            'habitat_risk': [],
            'uncertainty': []
        }
        
        current_input = x
        
        with torch.no_grad():
            for year in range(years):
                # Predict one year
                preds = self.forward(current_input, forecast_steps=samples_per_year)
                
                all_predictions['ice_maps'].append(preds['ice_maps'])
                all_predictions['habitat_risk'].append(preds['habitat_risk_class'])
                
                if self.use_uncertainty:
                    all_predictions['uncertainty'].append(preds['uncertainty'])
                
                # Update input with predictions for next iteration
                # Use predicted maps as new input
                new_input = preds['ice_maps']
                current_input = torch.cat([current_input[:, samples_per_year:], new_input], dim=1)
        
        # Concatenate all years
        all_predictions['ice_maps'] = torch.cat(all_predictions['ice_maps'], dim=1)
        all_predictions['habitat_risk'] = torch.cat(all_predictions['habitat_risk'], dim=1)
        
        if self.use_uncertainty:
            all_predictions['uncertainty'] = torch.cat(all_predictions['uncertainty'], dim=1)
        
        return all_predictions


def create_model(config) -> AntarcticSeaIceForecaster:
    """Create model from config"""
    return AntarcticSeaIceForecaster(
        image_size=config.data_config.image_size if hasattr(config, 'data_config') else (256, 256),
        in_channels=1,
        cnn_channels=config.model_config.cnn_channels[1:] if hasattr(config, 'model_config') else (32, 64, 128, 256),
        convlstm_hidden=getattr(config.model_config, 'convlstm_hidden_dim', 128) if hasattr(config, 'model_config') else 128,
        convlstm_layers=getattr(config.model_config, 'convlstm_num_layers', 2) if hasattr(config, 'model_config') else 2,
        transformer_dim=getattr(config.model_config, 'transformer_dim', 256) if hasattr(config, 'model_config') else 256,
        transformer_heads=getattr(config.model_config, 'transformer_heads', 8) if hasattr(config, 'model_config') else 8,
        transformer_layers=getattr(config.model_config, 'transformer_layers', 4) if hasattr(config, 'model_config') else 4,
        use_uncertainty=getattr(config.model_config, 'use_uncertainty', True) if hasattr(config, 'model_config') else True,
    )


if __name__ == "__main__":
    # Test model
    print("Testing AntarcticSeaIceForecaster...")
    
    model = AntarcticSeaIceForecaster(
        image_size=(64, 64),
        cnn_channels=(16, 32, 64),
        convlstm_hidden=32,
        convlstm_layers=1,
        transformer_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        prediction_horizon=12
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 1, 64, 64)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x, forecast_steps=12)
    
    print(f"Output ice_maps shape: {output['ice_maps'].shape}")
    print(f"Output habitat_risk_class shape: {output['habitat_risk_class'].shape}")
    print(f"Output habitat_risk_map shape: {output['habitat_risk_map'].shape}")
    
    if model.use_uncertainty:
        print(f"Output uncertainty shape: {output['uncertainty'].shape}")
    
    print("\nModel test passed!")