"""
LLM Fine-tuning with MX Format Quantization — Multi-GPU Accelerated

- DistributedDataParallel (DDP)
- torch.cuda.amp 
- 按GPU数量自动缩放 batch size 和学习率
- 支持 torchrun 启动
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from pychop.mx_formats import convert_linear_to_mx
from tqdm import tqdm
import argparse

def setup_ddp():
    """初始化 DDP 进程组。由 torchrun 自动注入环境变量。"""
    dist.init_process_group(backend="nccl")   # NCCL 是多GPU最快的后端
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    """只让 rank-0 进程打印日志、保存模型。"""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# ─────────────────────────────────────────────
# Model load + MX quantization
# ─────────────────────────────────────────────

def setup_mx_model(model_name: str, mx_format: str = "mxfp8_e4m3", block_size: int = 32):
    if is_main_process():
        print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 转换为 MX 量化层（在移到 GPU 之前做，节省显存）
    model = convert_linear_to_mx(
        model,
        format=mx_format,
        block_size=block_size,
        quantize_input=True,
        quantize_output=False,
        inplace=True,
    )

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ MX({mx_format}) 转换完成，参数量: {n_params:,}")

    return model, tokenizer


# ─────────────────────────────────────────────
# 单步训练
# ─────────────────────────────────────────────
def train_step(model, batch, optimizer, scheduler, device, scaler):
    """
    使用 AMP（自动混合精度）+ MX 量化的单步训练。
    
    注意：MX 量化在自定义的前向/反向中已经处理了低精度；
    AMP 的 autocast 作用于其余算子（LayerNorm、Softmax 等），
    两者可以共存。
    """
    model.train()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = input_ids.clone()

    optimizer.zero_grad()

    # autocast 开启 BF16/FP16 混合精度（NCCL+BF16 最稳定）
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

    # GradScaler 负责 loss scale（防止 FP16 下溢）
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    return loss.item()


# ─────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            total_steps += 1

    # 跨所有 GPU 汇总 loss（DDP 下必须）
    loss_tensor = torch.tensor(total_loss / total_steps, device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

    return loss_tensor.item()


# ─────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────

def finetune_llm_with_mx(
    model_name: str = "gpt2",
    mx_format: str = "mxfp8_e4m3",
    block_size: int = 32,
    dataset_name: str = "wikitext",
    num_epochs: int = 3,
    batch_size: int = 8,       # 每张 GPU 上的 batch size
    learning_rate: float = 5e-5,
    max_length: int = 512,
    use_ddp: bool = True,
):
    if use_ddp and torch.cuda.device_count() > 1:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        # 单卡或 CPU fallback
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_ddp = False

    if is_main_process():
        print("=" * 80)
        print("LLM Fine-tuning with MX Format Quantization (Multi-GPU)")
        print("=" * 80)
        print(f"Model      : {model_name}")
        print(f"MX Format  : {mx_format}, block_size={block_size}")
        print(f"GPUs       : {world_size}")
        print(f"Batch/GPU  : {batch_size}  →  Global batch: {batch_size * world_size}")
        print("=" * 80)

    model, tokenizer = setup_mx_model(model_name, mx_format, block_size)
    model = model.to(device)

    if use_ddp:
        # find_unused_parameters=False 性能更好；若有未用参数才设 True
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ── 数据集 ────────────────────────────────
    from datasets import load_dataset

    if is_main_process():
        print(f"\nLoading dataset: {dataset_name}")

    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"]["text"][:1000]
        val_texts = dataset["validation"]["text"][:100]
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported in this demo")

    def tokenize_function(texts):
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    if is_main_process():
        print("Tokenizing dataset...")

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    train_dataset = SimpleDataset(train_encodings)
    val_dataset = SimpleDataset(val_encodings)

    # DDP 专用 Sampler —— 保证每张 GPU 拿到不同的数据分片
    train_sampler = DistributedSampler(train_dataset) if use_ddp else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),   # 使用 sampler 时 shuffle 必须为 False
        num_workers=4,
        pin_memory=True,                   # 加速 CPU→GPU 传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── 优化器 & Scheduler ────────────────────
    # 多卡时学习率线性缩放（Linear Scaling Rule）
    scaled_lr = learning_rate * world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # AMP GradScaler（BF16 不需要 scale，但保留兼容性）
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # BF16 不需要，FP16 改为 True

    # ── 训练循环 ─────────────────────────────��
    if is_main_process():
        print(f"\nStarting training: {num_epochs} epochs, "
              f"{total_steps} total steps (per GPU)")
        print("=" * 80)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # 每个 epoch 更新 sampler 的 seed，保证多卡 shuffle 不重复
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)

        model.train()
        train_loss = 0.0

        progress_bar = (
            tqdm(train_loader, desc="Training")
            if is_main_process()
            else train_loader
        )

        for batch in progress_bar:
            loss = train_step(model, batch, optimizer, scheduler, device, scaler)
            train_loss += loss
            if is_main_process():
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        if is_main_process():
            print("Evaluating...")
        val_loss = evaluate(model, val_loader, device)

        if is_main_process():
            print(f"Epoch {epoch + 1} Results:")
            print(f"  Train Loss : {avg_train_loss:.4f}")
            print(f"  Val Loss   : {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("  ✓ New best model! Saving...")
                # get DDP wrapped model for saving
                raw_model = model.module if use_ddp else model
                torch.save(raw_model.state_dict(), f"best_model_{mx_format}.pt")

    if is_main_process():
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 80)

    if use_ddp:
        cleanup_ddp()

    return model

def main():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning with MX Format (Multi-GPU)")
    parser.add_argument("--model",      type=str,   default="gpt2")
    parser.add_argument("--format",     type=str,   default="mxfp8_e4m3")
    parser.add_argument("--block-size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=8,
                        help="batch size for single GPU; global batch size = batch_size * num_gpus")
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--no-ddp",     action="store_true",
                        help="deprecated DDP")
    args = parser.parse_args()

    finetune_llm_with_mx(
        model_name=args.model,
        mx_format=args.format,
        block_size=args.block_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_ddp=not args.no_ddp,
    )


if __name__ == "__main__":
    main()