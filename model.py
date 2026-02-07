import torch
from torch import nn
import torch.nn.functional as F
import math
from functools import partial
import numpy as np

# Add device detection at the beginning of file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    raise RuntimeError("Error: Model requires CUDA support to run. Please ensure you have an available GPU in your environment.")

class SELayer(nn.Module):
    """Squeeze-and-Excitation attention layer"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for reduced parameters"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = F.relu6(x)
        x = self.pointwise(x)
        return x

class FractalConv(nn.Module):
    """
    Convolution module combining fractal concepts to capture multi-scale features through recursive structure
    """
    
    def __init__(
        self,
        d_model,
        d_state=16,
        fractal_depth=3,
        dropout=0.,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.fractal_depth = fractal_depth
        
        # Input projection
        self.in_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        
        # Gating mechanism
        self.gate_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        
        # Fractal convolution layers - capture information at different scales across levels
        self.fractal_convs = nn.ModuleList()
        for i in range(fractal_depth):
            # Dilation rate increases at each level, exponentially expanding receptive field
            dilation = 2**i
            self.fractal_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        d_model, d_model, 
                        kernel_size=3, 
                        padding=dilation, 
                        dilation=dilation, 
                        groups=d_model
                    ),
                    nn.BatchNorm2d(d_model),
                    nn.GELU()
                )
            )
        
        # Fractal composition layer - learn how to combine information at different scales
        self.mix_weights = nn.Parameter(torch.ones(fractal_depth, d_model) / fractal_depth)
        
        # 1D fractal mixing layer - capture long-range dependencies in sequences
        self.seq_mixers = nn.ModuleList()
        for i in range(fractal_depth):
            kernel_size = 2**(i+1) + 1 # 3, 5, 9, 17...
            padding = kernel_size // 2
            self.seq_mixers.append(
                nn.Conv1d(
                    d_model, d_model, 
                    kernel_size=kernel_size, 
                    padding=padding, 
                    groups=d_model
                )
            )
        
        # Channel mixing layer - apply non-linear mapping on fractal features
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, kernel_size=1),
            nn.BatchNorm2d(d_model * 2),
            nn.GELU(),
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
    def forward(self, x):
        """
        Input: [B, C, H, W]
        Output: [B, C, H, W]
        """
        identity = x
        
        # Input projection
        x_in = self.in_proj(x)
        
        # Apply fractal convolution - capture multi-scale spatial features
        fractal_outputs = []
        for conv in self.fractal_convs:
            fractal_outputs.append(conv(x_in))
        
        # Merge fractal convolution results using learned weights
        mix_weights = F.softmax(self.mix_weights, dim=0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, D, C, 1, 1]
        spatial_out = sum([out * mix_weights[:, i] for i, out in enumerate(fractal_outputs)])
        
        # Gating mechanism
        gated = spatial_out * torch.sigmoid(self.gate_proj(x))
        
        # Save original shape
        B, C, H, W = gated.shape
        
        # Reshape to sequence form to handle sequence dependencies
        x_seq = gated.reshape(B, C, H*W) # [B, C, H*W]
        
        # Apply sequence fractal mixer - capture long-range dependencies in sequences
        seq_outputs = []
        for mixer in self.seq_mixers:
            seq_outputs.append(mixer(x_seq))
        
        # Merge sequence fractal results (using same mixing weights but on sequence dimension)
        mix_weights_seq = F.softmax(self.mix_weights, dim=0).unsqueeze(0).unsqueeze(-1) # [1, D, C, 1]
        seq_out = sum([out * mix_weights_seq[:, i] for i, out in enumerate(seq_outputs)])
        
        # Reshape back to 2D form
        out = seq_out.reshape(B, C, H, W)
        
        # Apply channel mixer
        out = self.channel_mixer(out)
        
        # Residual connection
        out = out + identity
        
        return self.dropout(out)

class DualBranchBlock(nn.Module):
    """Dual-branch mixing block combining fractal concepts"""
    
    def __init__(
        self,
        dim,
        drop_path=0.,
        d_state=16,
        fractal_depth=3,
        **kwargs,
    ):
        super().__init__()
        
        # Ensure number of channels is divisible by 2
        self.dim = dim
        self.dim_half = dim // 2
        
        # Fractal convolution path - using fractal convolution module
        self.fractal_path = nn.Sequential(
            nn.BatchNorm2d(self.dim_half),
            FractalConv(
                d_model=self.dim_half,
                d_state=d_state,
                fractal_depth=fractal_depth,
                dropout=drop_path
            )
        )
        
        # Convolution path
        self.conv_path = nn.Sequential(
            nn.BatchNorm2d(self.dim_half),
            nn.Conv2d(self.dim_half, self.dim_half, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dim_half),
            nn.ReLU(),
            nn.Conv2d(self.dim_half, self.dim_half, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dim_half),
            nn.ReLU(),
            nn.Conv2d(self.dim_half, self.dim_half, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Input x shape is [B, C, H, W]
        """
        # Check if input channel count matches expected
        actual_channels = x.shape[1]
        if actual_channels != self.dim:
            raise ValueError(f"Input channel count {actual_channels} does not match expected channel count {self.dim} ")
        
        # Ensure number of channels is divisible by 2
        if actual_channels % 2 != 0:
            pad = torch.zeros_like(x[:, :1])
            x = torch.cat([x, pad], dim=1)
            print(f"Warning: Channel count {actual_channels} is not divisible by 2, padding added")
        
        # Split into equal parts along channel dimension
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Pass through two paths separately
        x1 = self.conv_path(x1)
        x2 = self.fractal_path(x2)
        
        # Concatenate two outputs
        out = torch.cat([x1, x2], dim=1)
        
        # Residual connection
        if out.shape[1] == x.shape[1]:
            return out + x
        else:
            # Truncate to match original input size
            return out[:, :x.shape[1]] + x

class FractalMultiScaleBlock(nn.Module):
    """Fractal multi-scale block"""
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 fractal_level=0,
                 max_fractal_levels=3,
                 patch_size=2,
                 expand_ratio=1.5,
                 d_state=16):
        super(FractalMultiScaleBlock, self).__init__()

        self.fractal_level = fractal_level
        self.max_fractal_levels = max_fractal_levels
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.expand = max(8, int(in_channels * expand_ratio))
        if self.expand % 2 != 0:
            self.expand += 1
        self.use_res_connect = in_channels == out_channels

        # 1x1 Convolution layer - increase channel dimension
        self.conv1 = nn.Conv2d(in_channels, self.expand, kernel_size=1, bias=False)
        
        # Hybrid block
        self.hybrid_block = DualBranchBlock(
            dim=self.expand, 
            d_state=d_state,
        )
        
        # 1x1 Convolution layer - decrease channel dimension
        self.conv2 = nn.Conv2d(self.expand, out_channels, kernel_size=1, bias=False)
        
        # SE attention layer
        self.se = SELayer(out_channels, reduction=16)

        if fractal_level < max_fractal_levels - 1:
            # Downsample layer
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=patch_size,
                         stride=patch_size, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

            # Next fractal level
            self.next_fractal = FractalMultiScaleBlock(
                out_channels,
                out_channels * 2,
                fractal_level=fractal_level + 1,
                max_fractal_levels=max_fractal_levels,
                patch_size=patch_size,
                expand_ratio=expand_ratio,
                d_state=d_state
            )

            # Upsample layer
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channels * 2,
                    out_channels,
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

        # Move model to GPU in __init__ method
        self.to(device)

    def forward(self, x):
        identity = x

        # Expand channels
        x = self.conv1(x)
        x = F.relu6(x)
        
        # Hybrid block
        x = self.hybrid_block(x)
        
        # Reduce channels
        x = self.conv2(x)

        # SE attention
        x = self.se(x)

        # Residual connection (if dimensions match)
        if self.use_res_connect and x.shape == identity.shape:
            x = x + identity

        current_features = x

        # Multi-scale fusion
        if hasattr(self, 'next_fractal'):
            # Downsample current features
            down_features = self.downsample(current_features)

            # Process at next fractal level
            next_level_features = self.next_fractal(down_features)

            # Upsample back to current resolution
            up_features = self.upsample(next_level_features)

            # Resize if necessary
            if up_features.shape[-2:] != current_features.shape[-2:]:
                up_features = F.interpolate(
                    up_features,
                    size=current_features.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Fusion
            x = torch.cat([current_features, up_features], dim=1)
            x = self.fusion(x)

        return x

class SelfReflectionModule(nn.Module):
    """Reflection module for self-reflection mechanism"""

    def __init__(self, channels):
        super(SelfReflectionModule, self).__init__()

        self.reflection_layer = nn.Sequential(
            nn.Conv2d(channels, max(2, channels // 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(2, channels // 8), 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        reflection = self.reflection_layer(x)
        return x, reflection

class FractalMultiScaleUNetABL(nn.Module):
    """
    Fractal Multi-Scale UNet with Abduction learning
    """
    
    def __init__(self,
                 in_channels=3,
                 out_channels=2,
                 base_channels=6,
                 fractal_depth=3,
                 patch_size=2,
                 expand_ratio=1.5,
                 d_state=16,
                 reflection_threshold=0.5):
        super(FractalMultiScaleUNetABL, self).__init__()
        
        self.reflection_threshold = reflection_threshold
        
        # Initial convolution layer
        self.init_conv = DepthwiseSeparableConv(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        
        # Fractal generator
        self.fractal_generator = FractalMultiScaleBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            fractal_level=0,
            max_fractal_levels=fractal_depth,
            patch_size=patch_size,
            expand_ratio=expand_ratio,
            d_state=d_state
        )
        
        # Output from fractal_generator
        self.intuitive_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Reflection module
        self.reflection_module = SelfReflectionModule(base_channels)
        
        # Final output (straight from fractal features)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Aggregation layer for reflection
        self.reflection_aggregator = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Move model to GPU in __init__ method
        self.to(device)

    def forward(self, x, use_fusion=False):
        # Initial convolution
        x = self.init_conv(x)
        
        # Fractal generation
        fractal_features = self.fractal_generator(x)
        
        # Output from fractal_generator
        intuitive_output = self.intuitive_conv(fractal_features)
        
        # Reflection module
        features, reflection = self.reflection_module(fractal_features)
        
        # Aggregate reflection
        combined_reflection = self.reflection_aggregator(reflection)
        
        # Final output (straight from fractal features)
        final_output = self.final_conv(features)
        
        # Fusion mode output
        if use_fusion:
            fused_output = self.confidence_fusion(intuitive_output, final_output, combined_reflection)
            return fused_output, combined_reflection, intuitive_output, final_output
        
        return intuitive_output, combined_reflection

    def confidence_fusion(self, intuitive_output, final_output, reflection):
        """
        Fusion mechanism using reflection confidence
        
        Args:
            intuitive_output: Output from FractalMultiScaleBlock [B, C, H, W]
            final_output: [B, C, H, W]
            reflection: Reflection vector/confidence [B, 1, H, W]
        
        Returns:
            Fused output [B, C, H, W]
        """
        # Calculate confidence mask
        confidence_mask = (reflection >= self.reflection_threshold).float()
        
        # Fusion: use intuitive when confident, otherwise use final
        fused_output = intuitive_output * confidence_mask + final_output * (1 - confidence_mask)
        
        return fused_output
    
    def set_knowledge_base(self, knowledge_base):
        """Set knowledge base for abduction learning"""
        self.knowledge_base = knowledge_base

    def apply_abduction(self, intuitive_output, reflection_vector, knowledge_base_fn, final_output=None):
        """
        Apply abduction learning mechanism (Fusion mode)

        Args:
            intuitive_output: Intuitive output (from FractalMultiScaleBlock)
            reflection_vector: Reflection vector, confidence score
            knowledge_base_fn: Knowledge base function
            final_output: Final output (optional)

        Returns:
            Corrected output
        """

        mask = (reflection_vector < self.reflection_threshold).float()

        # If final_output exists, use Fusion mode
        if final_output is not None:
            # Combine intuitive_output and final_output
            confidence_mask = (reflection_vector >= self.reflection_threshold).float()
            fused_output = intuitive_output * confidence_mask + final_output * (1 - confidence_mask)
            
            # Apply knowledge base correction
            corrected_output = knowledge_base_fn(fused_output, mask)
        else:
            # If no final_output: apply knowledge base directly
            corrected_output = knowledge_base_fn(intuitive_output, mask)

        return corrected_output

# Parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# FLOPs
def count_flops(model, input_size=(1, 3, 224, 224)):
    """
    Count model FLOPs (floating point operations)
    
    Args:
        model: Model to count FLOPs for
        input_size: Input tensor size, default (1, 3, 224, 224)
    
    Returns:
        FLOPs count (in millions)
    """
    try:
        from thop import profile
        
        input = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(input,), verbose=False)
        return flops / 1e6 # FLOPs in millions
    except ImportError:
        print("Please install thop library for FLOPs: pip install thop")
        return None

if __name__ == "__main__":
    # Create model
    model = FractalMultiScaleUNetABL(
        in_channels=3, 
        out_channels=2, 
        base_channels=6, 
        fractal_depth=3,
        d_state=16
    ).to(device) # Move to GPU
    
    # Set model to evaluation mode
    model.eval()
    
    # Count Parameters
    param_count = count_parameters(model)
    print(f"FractalMultiScaleUNetABL Parameters: {param_count:,}")
    
    # Count FLOPs
    flops = count_flops(model)
    if flops is not None:
        print(f"FractalMultiScaleUNetABL FLOPs: {flops:.2f}M")
    
    # Detailed FLOPs analysis
    try:
        from thop import profile
        from thop import clever_format
        
        print("\n===== Start detailed FLOPs analysis =====")
        
        # FLOPs analysis for different input sizes
        input_sizes = [(1, 3, 256, 256), (1, 3, 512, 512)]
        
        print("\nFLOPs analysis for different input sizes:")
        for size in input_sizes:
            print(f"Analyzing input size {size[2]}x{size[3]}...")
            dummy_input = torch.randn(size).to(device)
            with torch.no_grad(): # Disable gradient calculation
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                flops_readable, params_readable = clever_format([flops, params], "%.3f")
                print(f" {size[2]}x{size[3]}: FLOPs={flops_readable}, Parameters={params_readable}")
        
        # FLOPs proportion analysis
        print("\nFLOPs proportion analysis for model components:")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            # Initial convolution layer
            print("Analyzing initial convolution layer...")
            init_flops, _ = profile(model.init_conv, inputs=(dummy_input,), verbose=False)
            
            # Fractal generator
            features = model.init_conv(dummy_input)
            print("Analyzing fractal generator...")
            fractal_flops, _ = profile(model.fractal_generator, inputs=(features,), verbose=False)
            
            # Reflection module
            features = model.fractal_generator(features)
            print("Analyzing reflection module...")
            reflection_flops, _ = profile(model.reflection_module, inputs=(features,), verbose=False)
            
            # Final convolution layer
            features, _ = model.reflection_module(features)
            print("Analyzing final convolution layer...")
            final_flops, _ = profile(model.final_conv, inputs=(features,), verbose=False)
            
            total_flops = init_flops + fractal_flops + reflection_flops + final_flops
            
            # Format output
            init_flops_readable, fractal_flops_readable, reflection_flops_readable, final_flops_readable, total_flops_readable = clever_format(
                [init_flops, fractal_flops, reflection_flops, final_flops, total_flops], "%.3f")
            
            print(f"Initial convolution layer: {init_flops_readable} ({init_flops/total_flops*100:.2f}%)")
            print(f"Fractal generator: {fractal_flops_readable} ({fractal_flops/total_flops*100:.2f}%)")
            print(f"Reflection module: {reflection_flops_readable} ({reflection_flops/total_flops*100:.2f}%)")
            print(f"Final convolution layer: {final_flops_readable} ({final_flops/total_flops*100:.2f}%)")
            print(f"Total: {total_flops_readable}")
            print("\n===== Detailed FLOPs analysis complete =====")
        
    except ImportError as e:
        print(f"Cannot perform detailed FLOPs analysis: {e}")
        print("Please install thop library: pip install thop")
    except Exception as e:
        print(f"Error during FLOPs analysis: {e}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device) # Move to GPU
    with torch.no_grad():
        output, reflection = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Reflection vector: {reflection.shape}")
