#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.output)
    print(f"merged adapter into {args.output}")


if __name__ == "__main__":
    main()
