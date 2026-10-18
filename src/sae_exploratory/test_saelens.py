"""
测试SAELens安装和API
"""
import sys
import torch

print("Testing SAELens installation...")
print("="*50)

try:
    from sae_lens import SAE
    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens.saes import TrainingSAE, StandardTrainingSAE, StandardTrainingSAEConfig
    print("✓ SAELens imports successful")
except ImportError as e:
    print(f"✗ SAELens import failed: {e}")
    print("\nPlease install SAELens:")
    print("  pip install sae-lens")
    sys.exit(1)

print("\nTesting StandardTrainingSAEConfig...")
print("-"*50)

try:
    # 测试配置创建
    sae_cfg = StandardTrainingSAEConfig(
        d_in=512,
        d_sae=4096,
        l1_coefficient=0.001,
        dtype="float32",
        device="cpu",
        apply_b_dec_to_input=True,
        normalize_activations="none",
    )
    print("✓ StandardTrainingSAEConfig created successfully")
    print(f"  d_in: {sae_cfg.d_in}")
    print(f"  d_sae: {sae_cfg.d_sae}")
    print(f"  l1_coefficient: {sae_cfg.l1_coefficient}")
except Exception as e:
    print(f"✗ StandardTrainingSAEConfig creation failed: {e}")
    print("\nTrying to inspect StandardTrainingSAEConfig signature...")
    import inspect
    sig = inspect.signature(StandardTrainingSAEConfig.__init__)
    print(f"\nStandardTrainingSAEConfig parameters:")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            default = param.default
            if default == inspect.Parameter.empty:
                print(f"  {param_name}: (required)")
            else:
                print(f"  {param_name}: (default={default})")
    sys.exit(1)

print("\nTesting StandardTrainingSAE...")
print("-"*50)

try:
    # 测试SAE创建
    sae = StandardTrainingSAE(sae_cfg)
    print("✓ StandardTrainingSAE created successfully")
    
    # 测试前向传播
    test_input = torch.randn(4, 512)
    with torch.no_grad():
        output = sae.forward(test_input)
    
    if isinstance(output, tuple):
        reconstructed, features = output
        print(f"✓ Forward pass successful (tuple output)")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Features shape: {features.shape}")
    else:
        print(f"✓ Forward pass successful (single output)")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        
except Exception as e:
    print(f"✗ StandardTrainingSAE test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("All tests passed! SAELens is working correctly.")
print("="*50)




