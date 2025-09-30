#!/usr/bin/env python3
"""
Layer-specific Feature Extractor for OPERA-CT
Extracts features from different layers of the OPERA-CT model
"""

import sys
import os
import numpy as np
import torch
import tempfile
import soundfile as sf
from pathlib import Path

def setup_opera_environment():
    """Set up OPERA-CT environment."""
    opera_path = Path.cwd() / "setup" / "OPERA"
    
    if str(opera_path) not in sys.path:
        sys.path.append(str(opera_path))
    
    os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:{opera_path}"
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    return opera_path

class LayerSpecificExtractor:
    """Extract features from specific OPERA-CT layers."""
    
    def __init__(self, layer_name="final"):
        self.layer_name = layer_name
        self.model = None
        self.hooks = []
        self.layer_outputs = {}
        
        # Set up OPERA environment
        setup_opera_environment()
        
        # Initialize model
        self._initialize_model()
        
        # Set up layer hooks
        self._setup_layer_hooks()
    
    def _initialize_model(self):
        """Initialize OPERA-CT model."""
        try:
            from src.benchmark.model_util import initialize_pretrained_model, get_encoder_path
            
            # Load model
            self.model = initialize_pretrained_model("operaCT")
            self.model.eval()
            
            # Load weights
            encoder_path = get_encoder_path("operaCT")
            ckpt = torch.load(encoder_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
            
            print(f"✅ OPERA-CT model initialized for layer extraction")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OPERA-CT model: {e}")
    
    def _setup_layer_hooks(self):
        """Set up forward hooks to capture layer outputs."""
        
        # Define layer mapping
        layer_mappings = {
            "layer_0": "encoder.encoder.htsat.layers.0",
            "layer_1": "encoder.encoder.htsat.layers.1", 
            "layer_2": "encoder.encoder.htsat.layers.2",
            "layer_3": "encoder.encoder.htsat.layers.3",
            "final": None  # Use standard extract_feature method
        }
        
        if self.layer_name == "final":
            # Use standard method, no hooks needed
            return
        
        target_layer_name = layer_mappings.get(self.layer_name)
        if not target_layer_name:
            raise ValueError(f"Unknown layer: {self.layer_name}")
        
        # Find target module
        target_module = None
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Layer not found: {target_layer_name}")
        
        # Register hook
        def hook_fn(module, input, output):
            self.layer_outputs[self.layer_name] = output
        
        hook = target_module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        
        print(f"✅ Hook registered for layer: {target_layer_name}")
    
    def extract_features_from_layer(self, audio_segments, sr):
        """Extract features from specified layer."""
        
        if self.layer_name == "final":
            # Use standard OPERA-CT extraction
            return self._extract_final_layer_features(audio_segments, sr)
        else:
            # Use layer-specific extraction
            return self._extract_intermediate_layer_features(audio_segments, sr)
    
    def _extract_final_layer_features(self, audio_segments, sr):
        """Extract features using standard OPERA-CT method."""
        try:
            from src.benchmark.model_util import extract_opera_feature
            
            # Save segments as temporary files
            temp_files = []
            temp_dir = Path(tempfile.mkdtemp())
            
            for i, segment in enumerate(audio_segments):
                temp_path = temp_dir / f"segment_{i:04d}.wav"
                sf.write(temp_path, segment, sr)
                temp_files.append(str(temp_path))
            
            # Extract features
            features = extract_opera_feature(temp_files, pretrain="operaCT", input_sec=len(audio_segments[0])/sr, dim=768)
            
            # Clean up
            for temp_file in temp_files:
                os.remove(temp_file)
            temp_dir.rmdir()
            
            return features, 768
            
        except Exception as e:
            return None, f"Final layer extraction failed: {e}"
    
    def _extract_intermediate_layer_features(self, audio_segments, sr):
        """Extract features from intermediate layers using hooks."""
        try:
            from src.util import pre_process_audio_mel_t
            
            features = []
            
            for segment in audio_segments:
                # Convert audio to OPERA-CT format (mel spectrogram)
                mel_spec = pre_process_audio_mel_t(segment, sample_rate=sr, n_mels=64, f_min=50, f_max=2000)
                
                # Prepare input tensor
                mel_tensor = torch.tensor(mel_spec, dtype=torch.float).unsqueeze(0)  # Add batch dimension
                
                # Clear previous outputs
                self.layer_outputs.clear()
                
                # Forward pass (will trigger hooks)
                with torch.no_grad():
                    try:
                        _ = self.model.extract_feature(mel_tensor, 768)  # Use extract_feature method
                    except Exception as e:
                        # Try alternative forward method
                        _ = self.model(mel_tensor)
                
                # Get layer output
                if self.layer_name in self.layer_outputs:
                    layer_output = self.layer_outputs[self.layer_name]
                    
                    # Pool the output to get fixed-size features
                    if len(layer_output.shape) > 2:
                        # If output has spatial dimensions, pool them
                        pooled_output = torch.mean(layer_output.view(layer_output.size(0), -1), dim=1)
                    else:
                        pooled_output = layer_output.squeeze(0)
                    
                    features.append(pooled_output.cpu().numpy())
                else:
                    raise RuntimeError(f"Layer output not captured for {self.layer_name}")
            
            features_array = np.array(features)
            feature_dim = features_array.shape[1]
            
            return features_array, feature_dim
            
        except Exception as e:
            return None, f"Intermediate layer extraction failed: {e}"
    
    def cleanup(self):
        """Remove hooks and clean up."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print(f"✅ Cleaned up hooks for {self.layer_name}")

def extract_features_by_layer(audio_segments, sr, layer_name):
    """Convenience function to extract features from a specific layer."""
    extractor = LayerSpecificExtractor(layer_name)
    
    try:
        features, feature_dim = extractor.extract_features_from_layer(audio_segments, sr)
        return features, feature_dim, None
    except Exception as e:
        return None, 0, str(e)
    finally:
        extractor.cleanup()
