#!/usr/bin/env python3
"""
Download Qwen model for local conveyance analysis.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_qwen():
    """Download Qwen3-30B-A3B-FP8 model."""
    model_name = "Qwen/Qwen3-30B-A3B-FP8"
    
    logger.info(f"Downloading {model_name}...")
    logger.info("This is a 30B parameter model quantized to FP8 - it will take some time and disk space")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info("Tokenizer downloaded successfully")
        
        # Download model
        logger.info("Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu",  # Just download, don't load to GPU yet
            trust_remote_code=True
        )
        logger.info("Model downloaded successfully")
        
        # Test basic functionality
        logger.info("Testing model...")
        test_input = "Analyze the conveyance potential of this text:"
        inputs = tokenizer(test_input, return_tensors="pt")
        logger.info("Model and tokenizer are working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.info("\nAlternative: You can use a smaller Qwen model for testing:")
        logger.info("- Qwen/Qwen2.5-7B-Instruct (7B params, easier to run)")
        logger.info("- Qwen/Qwen2.5-14B-Instruct (14B params, good balance)")
        logger.info("- Qwen/Qwen2.5-32B-Instruct (32B params, high quality)")
        return False

if __name__ == "__main__":
    success = download_qwen()
    if success:
        print("\nModel downloaded successfully!")
        print("You can now run process_to_hades.py with local conveyance analysis")
    else:
        print("\nModel download failed. Check the error messages above.")