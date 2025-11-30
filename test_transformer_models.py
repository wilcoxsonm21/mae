"""Test script to verify TransformerAE and TransformerMAE implementations."""

import torch
from models import TransformerAE, TransformerMAE

def test_transformer_ae():
    """Test TransformerAE instantiation and forward pass."""
    print("Testing TransformerAE...")

    # Model parameters
    input_dim = 1024  # 32x32 grayscale
    latent_dim = 128
    batch_size = 4

    # Create model
    model = TransformerAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        image_size=32,
        in_channels=1,
        patch_size=4,
        embed_dim=384,
        num_heads=6,
        encoder_depth=6,
        decoder_depth=3,
        mlp_ratio=4,
        dropout=0.1
    )

    print(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check outputs
    assert 'reconstruction' in output, "Missing 'reconstruction' in output"
    assert 'latent' in output, "Missing 'latent' in output"
    assert output['reconstruction'].shape == (batch_size, input_dim), \
        f"Wrong reconstruction shape: {output['reconstruction'].shape}"
    assert output['latent'].shape == (batch_size, latent_dim), \
        f"Wrong latent shape: {output['latent'].shape}"

    # Test encode/decode separately
    z = model.encode(x)
    assert z.shape == (batch_size, latent_dim), f"Wrong encode output shape: {z.shape}"

    x_recon = model.decode(z)
    assert x_recon.shape == (batch_size, input_dim), f"Wrong decode output shape: {x_recon.shape}"

    print("  ✓ TransformerAE test passed!")
    return True

def test_transformer_mae():
    """Test TransformerMAE instantiation and forward pass."""
    print("\nTesting TransformerMAE...")

    # Model parameters
    input_dim = 1024  # 32x32 grayscale
    latent_dim = 128
    batch_size = 4

    # Create model
    model = TransformerMAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        image_size=32,
        in_channels=1,
        patch_size=4,
        embed_dim=384,
        num_heads=6,
        encoder_depth=6,
        decoder_depth=3,
        mlp_ratio=4,
        dropout=0.1,
        mask_ratio=0.75
    )

    print(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # Forward pass (with automatic masking)
    with torch.no_grad():
        output = model(x)

    # Check outputs
    assert 'reconstruction' in output, "Missing 'reconstruction' in output"
    assert 'latent' in output, "Missing 'latent' in output"
    assert 'mask' in output, "Missing 'mask' in output"
    assert 'masked_input' in output, "Missing 'masked_input' in output"

    assert output['reconstruction'].shape == (batch_size, input_dim), \
        f"Wrong reconstruction shape: {output['reconstruction'].shape}"
    assert output['latent'].shape == (batch_size, latent_dim), \
        f"Wrong latent shape: {output['latent'].shape}"
    assert output['mask'].shape == (batch_size, input_dim), \
        f"Wrong mask shape: {output['mask'].shape}"

    # Test with provided mask
    mask = torch.rand(batch_size, input_dim) > 0.5
    mask = mask.float()

    with torch.no_grad():
        output2 = model(x, mask=mask)

    assert torch.equal(output2['mask'], mask), "Provided mask not used correctly"

    # Test encode/decode separately (without mask)
    z = model.encode(x)
    assert z.shape == (batch_size, latent_dim), f"Wrong encode output shape: {z.shape}"

    x_recon = model.decode(z)
    assert x_recon.shape == (batch_size, input_dim), f"Wrong decode output shape: {x_recon.shape}"

    print("  ✓ TransformerMAE test passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Transformer Model Implementations")
    print("=" * 60)

    try:
        test_transformer_ae()
        test_transformer_mae()

        print("\n" + "=" * 60)
        print("✓ All tests passed! Implementation is complete.")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
