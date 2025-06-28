#!/usr/bin/env python3
# Copyright (c) 2025 Veritas Aequitas Holdings LLC. All rights reserved.
# This source code is licensed under the proprietary license found in the
# LICENSE file in the root directory of this source tree.
#
# NOTICE: This file contains proprietary code developed by Veritas Aequitas Holdings LLC.
# Unauthorized use, reproduction, or distribution is strictly prohibited.
# For inquiries, contact: contact@veritasandaequitas.com

import sys
from core.pipeline import run_trading_pipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <TICKER>")
        sys.exit(1)
    ticker = sys.argv[1].strip().upper()
    result = run_trading_pipeline(ticker)
    print(result)

if __name__ == "__main__":
    main()
