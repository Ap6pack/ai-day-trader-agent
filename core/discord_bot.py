#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import discord
from discord.ext import commands
from config import settings
from core.pipeline import run_trading_pipeline

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.command()
async def trade(ctx, ticker: str):
    await ctx.send(f"Processing trade analysis for: {ticker.upper()}...")
    try:
        # TODO: For true async, refactor pipeline and all API calls to use asyncio/aiohttp.
        result = run_trading_pipeline(ticker.upper())
        await ctx.send(result)
    except Exception as e:
        await ctx.send(f"❌ An unexpected error occurred: {str(e)}")

bot.run(settings.DISCORD_BOT_TOKEN)
