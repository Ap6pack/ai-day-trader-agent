#!/usr/bin/env python3
"""
Enhanced Discord bot interface for the AI Day Trader Agent.
Now supports dividend capture strategies and detailed trade analysis.
"""

import discord
from discord.ext import commands
import logging
from datetime import datetime
from config.env_loader import load_env_variables
from core.pipeline import run_enhanced_analysis
from core.dividend_database import DividendDatabase
from utils.logger import get_logger

# Set up logging
logger = get_logger('discord_bot')

# Load environment variables
env_vars = load_env_variables()
DISCORD_TOKEN = env_vars['DISCORD_BOT_TOKEN']

# Set up Discord bot with intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize database
db = DividendDatabase()


@bot.event
async def on_ready():
    """Log when bot is ready."""
    logger.info(f'{bot.user} has connected to Discord!')
    print(f'✅ {bot.user} is online and ready!')


@bot.command(name='trade', help='Analyze a stock ticker for trading opportunities')
async def trade_analysis(ctx, ticker: str = None):
    """
    Perform enhanced trading analysis on a given ticker.
    
    Usage: !trade APAM
    """
    if not ticker:
        await ctx.send("❌ Please provide a ticker symbol. Example: `!trade APAM`")
        return
    
    ticker = ticker.upper()
    
    # Send initial response
    embed = discord.Embed(
        title=f"🔍 Analyzing {ticker}...",
        description="Running multi-strategy analysis including dividend capture opportunities",
        color=discord.Color.blue()
    )
    message = await ctx.send(embed=embed)
    
    try:
        # Run enhanced analysis
        logger.info(f"Running analysis for {ticker} requested by {ctx.author}")
        result = run_enhanced_analysis(ticker, env_vars)
        
        # Create result embed
        if result['recommendation'] == 'BUY':
            color = discord.Color.green()
            emoji = "🟢"
        elif result['recommendation'] == 'SELL':
            color = discord.Color.red()
            emoji = "🔴"
        else:
            color = discord.Color.gold()
            emoji = "🟡"
        
        # Build detailed embed
        embed = discord.Embed(
            title=f"{emoji} {ticker} Analysis Complete",
            description=f"**Recommendation:** {result['recommendation']} {result['quantity']} shares",
            color=color,
            timestamp=datetime.utcnow()
        )
        
        # Add main analysis fields
        embed.add_field(
            name="📈 Confidence Level",
            value=result['confidence'],
            inline=True
        )
        
        embed.add_field(
            name="🎯 Primary Strategy",
            value=result['primary_strategy'].title(),
            inline=True
        )
        
        embed.add_field(
            name="📝 Reason",
            value=result['reason'],
            inline=False
        )
        
        # Add risk parameters if available
        if result.get('risk_parameters'):
            risk = result['risk_parameters']
            risk_text = f"Stop Loss: ${risk.get('stop_loss', 'N/A')}\n"
            risk_text += f"Take Profit: ${risk.get('take_profit', 'N/A')}\n"
            risk_text += f"Risk/Reward: {risk.get('risk_reward_ratio', 'N/A')}"
            
            embed.add_field(
                name="⚠️ Risk Management",
                value=risk_text,
                inline=True
            )
        
        # Add dividend info if it's a dividend capture
        if result.get('dividend_info'):
            div_info = result['dividend_info']
            div_text = f"Days to Ex-Dividend: {div_info['days_to_ex_dividend']}\n"
            div_text += f"Expected Dividend: ${div_info['expected_dividend']}"
            
            embed.add_field(
                name="💰 Dividend Information",
                value=div_text,
                inline=True
            )
        
        embed.set_footer(text=f"Requested by {ctx.author.name}")
        
        # Update the message with results
        await message.edit(embed=embed)
        
        # Log successful analysis
        logger.info(f"Analysis completed for {ticker}: {result['recommendation']}")
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        
        error_embed = discord.Embed(
            title="❌ Analysis Error",
            description=f"Failed to analyze {ticker}: {str(e)}",
            color=discord.Color.red()
        )
        await message.edit(embed=error_embed)


@bot.command(name='dividend', help='Check upcoming dividend capture opportunities')
async def dividend_opportunities(ctx, min_yield: float = 5.0):
    """
    Show upcoming dividend capture opportunities.
    
    Usage: !dividend 6.0 (shows stocks with >6% annual yield)
    """
    try:
        opportunities = db.get_dividend_capture_opportunities(
            min_yield=min_yield,
            days_ahead=14
        )
        
        if opportunities.empty:
            await ctx.send(f"No dividend opportunities found with yield > {min_yield}%")
            return
        
        embed = discord.Embed(
            title="💰 Upcoming Dividend Opportunities",
            description=f"Stocks with dividend yield > {min_yield}% in next 14 days",
            color=discord.Color.gold()
        )
        
        for _, opp in opportunities.head(5).iterrows():
            field_value = f"Yield: {opp['dividend_yield']:.2f}%\n"
            field_value += f"Ex-Date: {opp['ex_dividend_date']}\n"
            if opp['win_rate']:
                field_value += f"Historical Win Rate: {opp['win_rate']:.1f}%"
            
            embed.add_field(
                name=f"{opp['symbol']}",
                value=field_value,
                inline=True
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error fetching dividend opportunities: {str(e)}")
        await ctx.send(f"❌ Error fetching opportunities: {str(e)}")


@bot.command(name='performance', help='Show trading performance statistics')
async def show_performance(ctx, ticker: str = None):
    """
    Display performance statistics for a ticker or overall.
    
    Usage: !performance APAM
    """
    try:
        if ticker:
            ticker = ticker.upper()
            perf = db.get_trade_performance(ticker)
            title = f"📊 Performance for {ticker}"
        else:
            # Would need to implement overall performance
            await ctx.send("Overall performance tracking coming soon! Try: `!performance APAM`")
            return
        
        embed = discord.Embed(
            title=title,
            color=discord.Color.blue()
        )
        
        # Add performance metrics
        embed.add_field(
            name="Total Trades",
            value=str(perf['total_trades']),
            inline=True
        )
        
        embed.add_field(
            name="Win Rate",
            value=f"{perf['win_rate']:.1f}%",
            inline=True
        )
        
        embed.add_field(
            name="Profit Factor",
            value=f"{perf['profit_factor']:.2f}",
            inline=True
        )
        
        embed.add_field(
            name="Avg Win",
            value=f"{perf['average_win_pct']:.2f}%",
            inline=True
        )
        
        embed.add_field(
            name="Avg Loss",
            value=f"{perf['average_loss_pct']:.2f}%",
            inline=True
        )
        
        embed.add_field(
            name="Total P&L",
            value=f"${perf['total_profit_loss']:.2f}",
            inline=True
        )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Error fetching performance: {str(e)}")
        await ctx.send(f"❌ Error fetching performance: {str(e)}")


# Run the bot
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
