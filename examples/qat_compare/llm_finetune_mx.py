"""
LLM Fine-tuning with MX Format Quantization

This example demonstrates how to fine-tune large language models
(like GPT-2, LLaMA) using MX format quantization with STE.

Benefits:
- Memory reduction during training
- Faster training with lower precision
- Minimal accuracy loss with QAT
- Compatible with HuggingFace Transformers
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from pychop.mx_formats import (
    convert_linear_to_mx,
    MXQuantizerSTE,
    MXLinear,
    MXAttention
)
from tqdm import tqdm
import argparse


def setup_mx_model(model_name: str, mx_format: str = 'mxfp8_e4m3', block_size: int = 32):
    """
    Load a model and convert it to use MX quantization.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g., 'gpt2', 'facebook/opt-125m')
    mx_format : str
        MX format ('mxfp8_e4m3', 'mxfp6_e3m2', etc.)
    block_size : int
        Block size for MX quantization
    
    Returns
    -------
    model : nn.Module
        Model with MX quantized layers
    tokenizer : Tokenizer
        Tokenizer for the model
    """
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Converting to MX format: {mx_format}, block_size={block_size}")
    
    # Convert all Linear layers to MX quantized versions
    model = convert_linear_to_mx(
        model,
        format=mx_format,
        block_size=block_size,
        quantize_input=True,
        quantize_output=False,
        inplace=True
    )
    
    print(f"✓ Model converted! Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def train_step(model, batch, optimizer, scheduler, device):
    """Single training step with MX quantization."""
    model.train()
    
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = input_ids.clone()
    
    # Forward pass (with MX quantization + STE)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    
    # Backward pass (gradients flow through STE)
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    scheduler.step()
    
    return loss.item()


def evaluate(model, dataloader, device):
    """Evaluate model with MX quantization."""
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            total_steps += 1
    
    return total_loss / total_steps


def finetune_llm_with_mx(
    model_name: str = 'gpt2',
    mx_format: str = 'mxfp8_e4m3',
    block_size: int = 32,
    dataset_name: str = 'wikitext',
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Fine-tune LLM with MX format quantization.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model name
    mx_format : str
        MX format specification
    block_size : int
        Block size for MX quantization
    dataset_name : str
        Dataset to use for fine-tuning
    num_epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    learning_rate : float
        Learning rate
    max_length : int
        Maximum sequence length
    device : str
        Device to use ('cuda' or 'cpu')
    """
    print("="*80)
    print("LLM Fine-tuning with MX Format Quantization")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"MX Format: {mx_format}, Block Size: {block_size}")
    print(f"Device: {device}")
    print("="*80)
    
    # Setup model with MX quantization
    model, tokenizer = setup_mx_model(model_name, mx_format, block_size)
    model = model.to(device)
    
    # Load dataset (simplified for demo)
    # In practice, use datasets library or your custom dataset
    from datasets import load_dataset
    
    print(f"\nLoading dataset: {dataset_name}")
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        train_texts = dataset['train']['text'][:1000]  # Subset for demo
        val_texts = dataset['validation']['text'][:100]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported in this demo")
    
    # Tokenize
    def tokenize_function(texts):
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    print("Tokenizing dataset...")
    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)
    
    # Create dataloaders
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings['input_ids'])
        
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}
    
    train_dataset = SimpleDataset(train_encodings)
    val_dataset = SimpleDataset(val_encodings)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Total training steps: {total_steps}")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-"*80)
        
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        for batch in progress_bar:
            loss = train_step(model, batch, optimizer, scheduler, device)
            train_loss += loss
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        print("Evaluating...")
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ New best model! Saving...")
            torch.save(model.state_dict(), f'best_model_{mx_format}.pt')
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80)
    
    return model


def compare_mx_formats():
    """
    Compare different MX formats on LLM fine-tuning.
    
    This helps you choose the optimal format for your use case.
    """
    print("="*80)
    print("Comparing MX Formats for LLM Fine-tuning")
    print("="*80)
    
    formats_to_test = [
        ('mxfp8_e4m3', 32, "Standard 8-bit"),
        ('mxfp8_e5m2', 32, "8-bit with more range"),
        ('mxfp6_e3m2', 32, "6-bit balanced"),
        ((4, 3), 32, "Custom E4M3 (8-bit)"),
        ((5, 4), 64, "Custom E5M4 (10-bit)"),
    ]
    
    model_name = 'gpt2'  # Small model for quick comparison
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = []
    
    for format_spec, block_size, description in formats_to_test:
        print(f"\nTesting: {description}")
        print("-"*80)
        
        try:
            # Setup model
            model, tokenizer = setup_mx_model(model_name, format_spec, block_size)
            model = model.to(device)
            
            # Measure memory
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # Forward pass
            input_ids = torch.randint(0, 50257, (4, 128)).to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            
            if device == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            else:
                peak_memory = -1
            
            # Calculate compression
            total_params = sum(p.numel() for p in model.parameters())
            
            results.append({
                'format': description,
                'params': total_params,
                'memory_gb': peak_memory,
                'success': True
            })
            
            print(f"✓ Success!")
            print(f"  Parameters: {total_params:,}")
            if peak_memory > 0:
                print(f"  Peak memory: {peak_memory:.2f} GB")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                'format': description,
                'success': False,
                'error': str(e)
            })
        
        # Cleanup
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"{'Format':<30} {'Success':<10} {'Memory (GB)':<15}")
    print("-"*80)
    
    for result in results:
        success = "✓" if result['success'] else "✗"
        memory = f"{result.get('memory_gb', -1):.2f}" if result['success'] else "N/A"
        print(f"{result['format']:<30} {success:<10} {memory:<15}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='LLM Fine-tuning with MX Format')
    parser.add_argument('--model', type=str, default='gpt2',
                        help='Model name from HuggingFace')
    parser.add_argument('--format', type=str, default='mxfp8_e4m3',
                        help='MX format (mxfp8_e4m3, mxfp6_e3m2, etc.)')
    parser.add_argument('--block-size', type=int, default=32,
                        help='Block size for MX quantization')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different MX formats')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_mx_formats()
    else:
        finetune_llm_with_mx(
            model_name=args.model,
            mx_format=args.format,
            block_size=args.block_size,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )


if __name__ == "__main__":
    main()