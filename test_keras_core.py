#!/usr/bin/env python3
"""
Test script for keras_core.py CTC implementation.
Tests the updated TensorFlow 2.x / Keras 3.x compatible code.
"""

import numpy as np
import tensorflow as tf
from captchaResolver.engine import get_captcha_type_list
from captchaResolver.keras_core import (
    ctc_label_dense_to_sparse,
    ctc_batch_cost,
    ctc_decode,
    CTCLayer
)


def test_ctc_label_dense_to_sparse():
    """Test dense to sparse label conversion."""
    print("\n" + "="*60)
    print("Testing ctc_label_dense_to_sparse...")
    print("="*60)
    
    # Create test data
    labels = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.int32)
    label_lengths = tf.constant([3, 2], dtype=tf.int32)
    
    sparse = ctc_label_dense_to_sparse(labels, label_lengths)
    
    print(f"Input labels shape: {labels.shape}")
    print(f"Label lengths: {label_lengths.numpy()}")
    print(f"Sparse tensor indices: {sparse.indices.shape}")
    print(f"Sparse tensor values: {sparse.values.numpy()}")
    print("✓ Dense to sparse conversion successful")
    return True


def test_ctc_batch_cost():
    """Test CTC batch cost computation."""
    print("\n" + "="*60)
    print("Testing ctc_batch_cost...")
    print("="*60)
    
    batch_size = 2
    time_steps = 10
    num_classes = 6  # 5 characters + 1 blank
    
    # Create random predictions (softmax probabilities)
    logits = tf.random.uniform((batch_size, time_steps, num_classes), minval=-1., maxval=1.)
    probs = tf.nn.softmax(logits, axis=-1)
    
    # Create test labels
    labels = tf.constant([[1, 2, 3, 0], [2, 1, 0, 0]], dtype=tf.int32)
    input_length = tf.constant([[time_steps], [time_steps]], dtype=tf.int32)
    label_length = tf.constant([[3], [2]], dtype=tf.int32)
    
    # Test with probabilities (default)
    loss_from_probs = ctc_batch_cost(labels, probs, input_length, label_length, from_logits=False)
    print(f"Loss from probabilities shape: {loss_from_probs.shape}")
    print(f"Loss values: {loss_from_probs.numpy().flatten()}")
    assert tf.reduce_all(tf.math.is_finite(loss_from_probs)), "Loss contains NaN or Inf"
    print("✓ CTC loss from probabilities computed successfully")
    
    # Test with logits
    loss_from_logits = ctc_batch_cost(labels, logits, input_length, label_length, from_logits=True)
    print(f"Loss from logits shape: {loss_from_logits.shape}")
    print(f"Loss values: {loss_from_logits.numpy().flatten()}")
    assert tf.reduce_all(tf.math.is_finite(loss_from_logits)), "Loss contains NaN or Inf"
    print("✓ CTC loss from logits computed successfully")
    
    return True


def test_ctc_decode():
    """Test CTC decoding (both greedy and beam search)."""
    print("\n" + "="*60)
    print("Testing ctc_decode...")
    print("="*60)
    
    batch_size = 2
    time_steps = 10
    num_classes = 6
    
    # Create random predictions
    logits = tf.random.uniform((batch_size, time_steps, num_classes), minval=-1., maxval=1.)
    probs = tf.nn.softmax(logits, axis=-1)
    input_length = np.array([time_steps, time_steps])
    
    # Test greedy decoding
    decoded_greedy, log_probs_greedy = ctc_decode(probs, input_length, greedy=True)
    print(f"Greedy decoded shape: {decoded_greedy[0].shape}")
    print(f"Greedy decoded values:\n{decoded_greedy[0].numpy()}")
    print("✓ Greedy decoding successful")
    
    # Test beam search decoding
    decoded_beam, log_probs_beam = ctc_decode(probs, input_length, greedy=False, beam_width=10, top_paths=1)
    print(f"Beam search decoded shape: {decoded_beam[0].shape}")
    print(f"Beam search decoded values:\n{decoded_beam[0].numpy()}")
    print("✓ Beam search decoding successful")
    
    return True


def test_ctc_layer():
    """Test CTCLayer integration."""
    print("\n" + "="*60)
    print("Testing CTCLayer...")
    print("="*60)
    
    batch_size = 2
    time_steps = 10
    num_classes = 6
    max_label_len = 4
    
    # Create test inputs
    y_pred = tf.random.uniform((batch_size, time_steps, num_classes))
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_true = tf.constant([[1, 2, 3, 0], [2, 1, 0, 0]], dtype=tf.float32)
    
    # Test layer with default settings (from_logits=False)
    layer = CTCLayer(from_logits=False, blank_index=-1, name="ctc_test")
    output = layer(y_true, y_pred)
    
    print(f"Layer output shape: {output.shape}")
    print(f"Layer losses: {layer.losses}")
    assert len(layer.losses) > 0, "Layer should have added loss"
    print("✓ CTCLayer forward pass successful")
    
    # Test serialization
    config = layer.get_config()
    print(f"Layer config: {config}")
    assert "from_logits" in config, "Config should contain from_logits"
    assert "blank_index" in config, "Config should contain blank_index"
    print("✓ CTCLayer serialization successful")
    
    return True


def test_end_to_end():
    """Test complete workflow: loss computation -> decoding."""
    print("\n" + "="*60)
    print("Testing end-to-end workflow...")
    print("="*60)
    
    batch_size = 4
    time_steps = 15
    num_classes = 11  # 10 digits + blank
    
    # Simulate model predictions
    logits = tf.random.uniform((batch_size, time_steps, num_classes), minval=-2., maxval=2.)
    probs = tf.nn.softmax(logits, axis=-1)
    
    # Create synthetic labels
    labels = tf.constant([
        [1, 2, 3, 4, 0, 0],
        [5, 6, 7, 0, 0, 0],
        [8, 9, 1, 2, 3, 0],
        [4, 5, 0, 0, 0, 0]
    ], dtype=tf.int32)
    
    input_length = tf.constant([[time_steps]] * batch_size, dtype=tf.int32)
    label_length = tf.constant([[4], [3], [5], [2]], dtype=tf.int32)
    
    # Compute loss
    loss = ctc_batch_cost(labels, probs, input_length, label_length, from_logits=False)
    print(f"Batch loss shape: {loss.shape}")
    print(f"Loss values: {loss.numpy().flatten()}")
    print(f"Mean loss: {tf.reduce_mean(loss).numpy():.4f}")
    
    # Decode predictions
    input_len_np = np.array([time_steps] * batch_size)
    decoded, _ = ctc_decode(probs, input_len_np, greedy=True)
    print(f"Decoded predictions shape: {decoded[0].shape}")
    print(f"Sample decoded sequences:\n{decoded[0].numpy()[:2]}")
    
    print("✓ End-to-end workflow successful")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" Testing TensorFlow/Keras CTC Implementation")
    print(" Updated for TF 2.20.0 and Keras 3.x compatibility")
    print("="*70)
    
    tests = [
        ("Dense to Sparse Conversion", test_ctc_label_dense_to_sparse),
        ("CTC Batch Cost", test_ctc_batch_cost),
        ("CTC Decode", test_ctc_decode),
        ("CTC Layer", test_ctc_layer),
        ("End-to-End Workflow", test_end_to_end),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"✗ Test failed with error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print(" Test Summary")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:12} | {test_name}")
        if error:
            print(f"             Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Code is ready for use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    captcha_id = 'default'
    backend = 'keras'
    train_data = get_captcha_type_list(backend=backend)[captcha_id].train_data
    train_data.shuffle_train_data(train_size=0.9)

