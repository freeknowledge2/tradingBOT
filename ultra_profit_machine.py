#!/usr/bin/env python3
"""
Ultra Profit Machine - Maximum Aggressive Trading for Maximum Profits
"""

import os
import asyncio
import json
import logging
import aiohttp
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class UltraProfitMachine:
    def __init__(self):
        # Load credentials
        self.mt5_login = int(os.getenv('MT5_LOGIN'))
        self.mt5_password = os.getenv('MT5_PASSWORD')
        self.mt5_server = os.getenv('MT5_SERVER')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # ULTRA AGGRESSIVE parameters for MAXIMUM PROFITS
        self.symbol = "EURUSD"
        self.base_lot_size = 0.1   # 10x higher volume!
        self.max_positions = 20    # More positions
        self.profit_target_pips = 2  # Tiny profits but frequent
        self.stop_loss_pips = 4     # Very tight stops
        
        # MAXIMUM SPEED settings
        self.confidence_threshold = 0.45  # Lower threshold for more trades
        self.min_trade_interval = 10     # 10 seconds between trades
        self.force_trade_interval = 60   # Force trade every minute
        
        # PROFIT MAXIMIZATION features
        self.martingale_multiplier = 1.5  # Increase lot size after loss
        self.max_martingale_level = 3     # Max 3 levels of martingale
        self.profit_boost_on_streak = True # Increase size on winning streak
        
        # Files
        self.forex_data_file = "c:\\Users\\Antosh\\OneDrive\\Desktop\\tradingBOT\\forexdata.json"
        self.ultra_log_file = "c:\\Users\\Antosh\\OneDrive\\Desktop\\tradingBOT\\ultra_profit_log.json"
        
        # Trading state
        self.last_trade_time = None
        self.trades_today = 0
        self.winning_streak = 0
        self.losing_streak = 0
        self.current_martingale_level = 0
        self.total_profit_today = 0
        
        # Initialize
        self.connect_mt5()
        self.init_log()
        
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        try:
            mt5.shutdown()
            
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            if not mt5.login(self.mt5_login, self.mt5_password, self.mt5_server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
            
            logger.info(f"💰 Ultra Profit Machine connected to {self.mt5_server}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def init_log(self):
        """Initialize ultra profit log"""
        if not os.path.exists(self.ultra_log_file):
            log_data = {
                "created": datetime.now().isoformat(),
                "trades": [],
                "performance": {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_profit": 0.0,
                    "max_winning_streak": 0,
                    "max_losing_streak": 0,
                    "martingale_trades": 0,
                    "forced_trades": 0,
                    "profit_per_hour": 0.0
                }
            }
            
            with open(self.ultra_log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
    
    def calculate_lot_size(self) -> float:
        """Calculate dynamic lot size for maximum profits"""
        try:
            base_size = self.base_lot_size
            
            # Martingale after losses
            if self.losing_streak > 0 and self.current_martingale_level < self.max_martingale_level:
                martingale_size = base_size * (self.martingale_multiplier ** self.losing_streak)
                logger.info(f"🎰 Martingale Level {self.losing_streak}: {martingale_size:.2f} lots")
                return min(martingale_size, 0.5)  # Max 0.5 lots
            
            # Boost on winning streak
            if self.winning_streak >= 3 and self.profit_boost_on_streak:
                boost_size = base_size * 1.5
                logger.info(f"🚀 Winning streak boost: {boost_size:.2f} lots")
                return boost_size
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error calculating lot size: {e}")
            return self.base_lot_size
    
    def should_force_trade(self) -> bool:
        """Check if we should force a trade for maximum activity"""
        if not self.last_trade_time:
            return True
        
        time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
        return time_since_last > self.force_trade_interval
    
    def get_open_positions(self) -> List[Dict]:
        """Get current open positions"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                enhanced_positions = []
                for pos in positions:
                    pos_dict = pos._asdict()
                    
                    # Calculate pips profit
                    if pos.type == 0:  # BUY
                        pips_profit = (pos.price_current - pos.price_open) * 10000
                    else:  # SELL
                        pips_profit = (pos.price_open - pos.price_current) * 10000
                    
                    pos_dict['pips_profit'] = pips_profit
                    pos_dict['seconds_open'] = datetime.now().timestamp() - pos.time
                    enhanced_positions.append(pos_dict)
                return enhanced_positions
            return []
        except:
            return []
    
    async def get_ultra_analysis(self, market_data: Dict[str, Any], force_trade: bool = False) -> Dict[str, Any]:
        """Get ultra-aggressive analysis for maximum profits"""
        try:
            prompt = self.create_ultra_prompt(market_data, force_trade)
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.openai_api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are an ULTRA-AGGRESSIVE profit-maximizing trader. Your ONLY goal is MAXIMUM PROFITS through HIGH-FREQUENCY trading.

💰 ULTRA PROFIT PHILOSOPHY:
1. TRADE CONSTANTLY - Every opportunity = profit
2. MICRO PROFITS - 2 pips is enough, trade often
3. MAXIMUM VOLUME - Use highest possible lot sizes
4. SPEED IS MONEY - Make decisions instantly
5. NEVER MISS OPPORTUNITIES - Trade any reasonable setup

🚀 ULTRA AGGRESSIVE SIGNALS (Trade Almost Everything):
- BUY: RSI > 35, ANY upward momentum, price movement up
- SELL: RSI < 65, ANY downward momentum, price movement down
- FORCE_BUY: When forced, find ANY bullish signal
- FORCE_SELL: When forced, find ANY bearish signal

⚡ CONFIDENCE LEVELS (Very Liberal):
- 0.5+: Good enough - TRADE IT!
- 0.4-0.49: Acceptable - TRADE IT!
- 0.3-0.39: Weak but tradeable - TRADE IT!
- Below 0.3: Only if absolutely forced

🎯 PROFIT TARGETS:
- Micro scalp: 2 pips (trade constantly)
- Quick profit: 3-4 pips
- Never hold losing positions > 4 pips

💡 TRADING MENTALITY:
- Every pip counts
- Volume = profits
- Speed = money
- Hesitation = lost profits

Response format (JSON only):
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "brief profit-focused reason",
    "urgency": "MAXIMUM" | "HIGH" | "MEDIUM",
    "profit_potential": "EXCELLENT" | "GOOD" | "ACCEPTABLE",
    "volume_recommendation": "MAXIMUM" | "HIGH" | "NORMAL"
}"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.6,  # Higher for more varied aggressive responses
                    "max_tokens": 500
                }
                
                async with session.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        gpt_response = result['choices'][0]['message']['content']
                        
                        # Clean and parse JSON
                        clean_response = gpt_response.strip()
                        if clean_response.startswith('```json'):
                            clean_response = clean_response[7:]
                        if clean_response.startswith('```'):
                            clean_response = clean_response[3:]
                        if clean_response.endswith('```'):
                            clean_response = clean_response[:-3]
                        
                        clean_response = clean_response.strip()
                        
                        try:
                            analysis = json.loads(clean_response)
                            return analysis
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON from GPT: {gpt_response}")
                            # Return aggressive fallback
                            return {
                                "action": "BUY" if force_trade else "HOLD",
                                "confidence": 0.5 if force_trade else 0.0,
                                "reasoning": "JSON error - aggressive fallback",
                                "urgency": "MAXIMUM" if force_trade else "LOW",
                                "profit_potential": "GOOD",
                                "volume_recommendation": "HIGH"
                            }
                    else:
                        return {"action": "HOLD", "confidence": 0.0, "reasoning": "API error"}
        
        except Exception as e:
            logger.error(f"Error in ultra analysis: {e}")
            return {"action": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def create_ultra_prompt(self, market_data: Dict[str, Any], force_trade: bool = False) -> str:
        """Create ultra-aggressive profit-focused prompt"""
        latest_data = market_data
        
        # Extract data
        current_price = latest_data.get('real_time_price', {}).get('price', 0)
        indicators = latest_data.get('technical_indicators', {})
        
        # Get indicator values
        rsi_value = float(indicators.get('RSI', {}).get('values', [{}])[0].get('rsi', 50))
        macd_data = indicators.get('MACD', {}).get('values', [{}])[0]
        macd_line = float(macd_data.get('macd', 0))
        macd_signal = float(macd_data.get('macd_signal', 0))
        
        ema_20 = float(indicators.get('EMA_20', {}).get('values', [{}])[0].get('ema', current_price))
        
        # Get positions and profit info
        positions = self.get_open_positions()
        account_info = mt5.account_info()
        current_profit = account_info.profit if account_info else 0
        
        force_text = "🚨 MAXIMUM PROFIT MODE - FIND ANY TRADE!" if force_trade else ""
        
        prompt = f"""
💰 ULTRA PROFIT ANALYSIS - EURUSD
=================================
{force_text}
Current Price: {current_price:.5f}
Time: {datetime.now().strftime('%H:%M:%S')}

💸 PROFIT STATUS:
================
Current Account P&L: ${current_profit:+.2f}
Trades Today: {self.trades_today}
Winning Streak: {self.winning_streak}
Losing Streak: {self.losing_streak}
Open Positions: {len(positions)}

⚡ MARKET INDICATORS:
====================
RSI: {rsi_value:.1f} {'[BULLISH ZONE]' if rsi_value > 35 else '[BEARISH ZONE]' if rsi_value < 65 else '[NEUTRAL]'}
MACD: {'BULLISH' if macd_line > macd_signal else 'BEARISH'} (Line: {macd_line:.6f} vs Signal: {macd_signal:.6f})
EMA(20): {ema_20:.5f} - Price {'ABOVE' if current_price > ema_20 else 'BELOW'} by {abs(current_price - ema_20) * 10000:.1f} pips

🚀 PROFIT OPPORTUNITY:
=====================
Target: 2 pips profit (MICRO SCALPING)
Volume: 0.1+ lots (MAXIMUM VOLUME)
Strategy: {'FORCE ANY PROFITABLE TRADE' if force_trade else 'Ultra-aggressive profit hunting'}
Mindset: EVERY PIP = MONEY!

💡 DECISION CRITERIA:
====================
- ANY upward movement = BUY opportunity
- ANY downward movement = SELL opportunity
- RSI > 35 = Consider BUY
- RSI < 65 = Consider SELL
- Don't overthink - TRADE FOR PROFITS!

MAKE AN ULTRA-AGGRESSIVE PROFIT DECISION!
Find ANY opportunity to make money!
"""
        
        return prompt
    
    def execute_ultra_trade(self, action: str, analysis: Dict[str, Any]) -> bool:
        """Execute ultra-aggressive trade for maximum profits"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return False
            
            # Calculate dynamic lot size
            lot_size = self.calculate_lot_size()
            
            # Boost volume based on analysis
            volume_rec = analysis.get('volume_recommendation', 'NORMAL')
            if volume_rec == 'MAXIMUM':
                lot_size *= 1.5
            elif volume_rec == 'HIGH':
                lot_size *= 1.2
            
            lot_size = round(min(lot_size, 0.5), 2)  # Max 0.5 lots for safety
            
            if action == "BUY":
                entry_price = tick.ask
                take_profit = entry_price + (self.profit_target_pips / 10000)
                stop_loss = entry_price - (self.stop_loss_pips / 10000)
                order_type = mt5.ORDER_TYPE_BUY
                
            else:  # SELL
                entry_price = tick.bid
                take_profit = entry_price - (self.profit_target_pips / 10000)
                stop_loss = entry_price + (self.stop_loss_pips / 10000)
                order_type = mt5.ORDER_TYPE_SELL
            
            # Prepare order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": order_type,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 10,
                "magic": 234005,
                "comment": f"Ultra {action}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Execute order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                urgency = analysis.get('urgency', 'MEDIUM')
                profit_potential = analysis.get('profit_potential', 'GOOD')
                
                logger.info(f"💰 ULTRA {action}: {lot_size} lots at {result.price:.5f}")
                logger.info(f"   Urgency: {urgency} | Potential: {profit_potential}")
                logger.info(f"   Target: +{self.profit_target_pips} pips | Stop: -{self.stop_loss_pips} pips")
                
                # Log trade
                self.log_ultra_trade(action, result.price, stop_loss, take_profit, result.order, lot_size, analysis)
                self.last_trade_time = datetime.now()
                self.trades_today += 1
                
                return True
            else:
                logger.error(f"❌ Ultra {action} failed: {result.retcode} - {result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing ultra trade: {e}")
            return False
    
    def close_losing_positions(self) -> bool:
        """Close positions losing more than 4 pips"""
        try:
            positions = self.get_open_positions()
            closed_count = 0
            
            for position in positions:
                if position['pips_profit'] < -3:  # Close at -3 pips (tighter)
                    if self.close_position(position):
                        closed_count += 1
                        self.losing_streak += 1
                        self.winning_streak = 0
            
            return closed_count > 0
            
        except Exception as e:
            logger.error(f"Error closing losing positions: {e}")
            return False
    
    def close_position(self, position: Dict) -> bool:
        """Close a specific position"""
        try:
            if position['type'] == 0:  # BUY position
                order_type = mt5.ORDER_TYPE_SELL
            else:  # SELL position
                order_type = mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position['volume'],
                "type": order_type,
                "position": position['ticket'],
                "deviation": 15,
                "magic": 234005,
                "comment": "Ultra Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Closed losing position: {position['pips_profit']:+.1f} pips")
                return True
            else:
                logger.error(f"❌ Failed to close position: {result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def log_ultra_trade(self, action: str, price: float, sl: float, tp: float, ticket: int, volume: float, analysis: Dict[str, Any]):
        """Log ultra profit trade"""
        try:
            with open(self.ultra_log_file, 'r') as f:
                log_data = json.load(f)
            
            trade_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "price": price,
                "stop_loss": sl,
                "take_profit": tp,
                "ticket": ticket,
                "volume": volume,
                "confidence": analysis.get('confidence', 0),
                "reasoning": analysis.get('reasoning', ''),
                "urgency": analysis.get('urgency', 'MEDIUM'),
                "profit_potential": analysis.get('profit_potential', 'GOOD'),
                "volume_recommendation": analysis.get('volume_recommendation', 'NORMAL'),
                "winning_streak": self.winning_streak,
                "losing_streak": self.losing_streak,
                "martingale_level": self.current_martingale_level
            }
            
            log_data["trades"].append(trade_entry)
            log_data["performance"]["total_trades"] += 1
            
            with open(self.ultra_log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error logging ultra trade: {e}")
    
    async def run_ultra_profit_analysis(self):
        """Main ultra profit function"""
        try:
            logger.info("💰 Starting Ultra Profit Analysis...")
            
            # Check if data file exists
            if not os.path.exists(self.forex_data_file):
                logger.error("No forex data file found")
                return
            
            # Load latest market data
            with open(self.forex_data_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                logger.error("No market data available")
                return
            
            latest_data = data[-1]
            
            # Close losing positions first
            self.close_losing_positions()
            
            # Check if we should force a trade
            force_trade = self.should_force_trade()
            if force_trade:
                logger.info("🚨 FORCING ULTRA PROFIT TRADE!")
            
            # Get AI analysis
            analysis = await self.get_ultra_analysis(latest_data, force_trade)
            
            action = analysis.get('action', 'HOLD')
            confidence = analysis.get('confidence', 0.0)
            urgency = analysis.get('urgency', 'LOW')
            profit_potential = analysis.get('profit_potential', 'POOR')
            
            logger.info(f"📊 Ultra Analysis: {action} | Confidence: {confidence:.2f} | Urgency: {urgency}")
            logger.info(f"   Profit Potential: {profit_potential}")
            logger.info(f"   Reason: {analysis.get('reasoning', 'No reason')}")
            
            # Ultra-aggressive trading logic
            should_trade = False
            
            if force_trade and action in ['BUY', 'SELL']:
                should_trade = True
                logger.info("🚨 FORCED ULTRA PROFIT TRADE!")
            elif confidence >= self.confidence_threshold and action in ['BUY', 'SELL']:
                should_trade = True
                logger.info(f"💰 PROFIT TRADE: {confidence:.2f} >= {self.confidence_threshold}")
            elif urgency in ["MAXIMUM", "HIGH"] and confidence >= 0.3:
                should_trade = True
                logger.info("⚡ HIGH URGENCY PROFIT TRADE!")
            elif profit_potential in ["EXCELLENT", "GOOD"] and confidence >= 0.35:
                should_trade = True
                logger.info("🎯 HIGH PROFIT POTENTIAL TRADE!")
            
            # Check position limits
            current_positions = len(self.get_open_positions())
            if current_positions >= self.max_positions:
                should_trade = False
                logger.info(f"🛑 Position limit reached: {current_positions}/{self.max_positions}")
            
            if should_trade:
                success = self.execute_ultra_trade(action, analysis)
                if success:
                    logger.info(f"💰 Ultra {action} executed for MAXIMUM PROFITS!")
                else:
                    logger.warning(f"❌ Ultra {action} execution failed")
            else:
                logger.info(f"💤 No ultra trade: Confidence {confidence:.2f} < {self.confidence_threshold}")
                
        except Exception as e:
            logger.error(f"Error in ultra profit analysis: {e}")

def main():
    """Main function for testing"""
    import asyncio
    
    trader = UltraProfitMachine()
    
    # Run the ultra profit analysis
    asyncio.run(trader.run_ultra_profit_analysis())

if __name__ == "__main__":
    main()