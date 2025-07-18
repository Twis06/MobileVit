#!/usr/bin/env python3
import os
import wandb
import logging

# Enable wandb debugging
# os.environ['WANDB_DEBUG'] = 'true'
# os.environ['WANDB_CORE_DEBUG'] = 'true'

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("Testing wandb connectivity...")

# Test 1: Check wandb status
print("\n=== Test 1: wandb status ===")
os.system("wandb status")

# Test 2: Test offline mode
print("\n=== Test 2: Offline mode ===")
try:
    print("Initializing wandb in offline mode...")
    run = wandb.init(project="test-project", mode="offline")
    print("✓ Offline mode successful")
    wandb.log({"test": 1})
    print("✓ Logging successful")
    wandb.finish()
    print("✓ Finished successfully")
except Exception as e:
    print(f"✗ Offline mode failed: {e}")

# Test 3: Test online mode
print("\n=== Test 3: Online mode ===")
try:
    print("Initializing wandb in online mode...")
    run = wandb.init(project="test-project", mode="online")
    print("✓ Online mode successful")
    wandb.log({"test": 1})
    print("✓ Logging successful")
    wandb.finish()
    print("✓ Finished successfully")
except Exception as e:
    print(f"✗ Online mode failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check network connectivity
print("\n=== Test 4: Network connectivity ===")
import subprocess
import sys

try:
    result = subprocess.run(['curl', '-I', 'https://api.wandb.ai'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✓ Network connectivity to wandb API successful")
        print(result.stdout)
    else:
        print("✗ Network connectivity failed")
        print(result.stderr)
except Exception as e:
    print(f"✗ Network test failed: {e}")

print("\n=== Test completed ===")
