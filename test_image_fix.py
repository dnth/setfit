#!/usr/bin/env python3
"""
Simple test script to verify the SetFitImageEncoder fix.
"""

import torch
from setfit import SetFitImageEncoder

def test_image_encoder():
    """Test that SetFitImageEncoder has the required methods."""
    print("Testing SetFitImageEncoder...")

    # Create an image encoder
    encoder = SetFitImageEncoder(model_name="resnet18")

    # Test that parameters() method exists and returns something
    try:
        params = list(encoder.parameters())
        print(f"✅ parameters() method works! Found {len(params)} parameter groups")
    except AttributeError as e:
        print(f"❌ parameters() method failed: {e}")
        return False

    # Test that train() method exists
    try:
        encoder.train(True)  # Should not raise an error
        print("✅ train() method works!")
    except AttributeError as e:
        print(f"❌ train() method failed: {e}")
        return False

    # Test that we can create a simple optimizer
    try:
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.001)
        print("✅ Can create optimizer with encoder parameters!")
    except Exception as e:
        print(f"❌ Failed to create optimizer: {e}")
        return False

    print("🎉 All tests passed! The fix works correctly.")
    return True

if __name__ == "__main__":
    test_image_encoder()