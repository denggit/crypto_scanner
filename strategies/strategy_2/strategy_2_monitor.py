#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/12/25 1:20 PM
@File       : strategy_2_monitor.py
@Description: 高频短线策略实盘模拟监控系统
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()

from strategies.strategy_2.methods.monitor import Strategy2Monitor
from strategies.strategy_2.methods.trader import Strategy2Trader
from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_2.strategy_2 import HighFrequencyStrategy
from utils.logger import logger


class StrategyMonitor(Strategy2Monitor):
    """Strategy Monitor with Unified Logic for Mock and Real Trading"""

    def __init__(self, symbol: str, bar: str = '1m',
                 consecutive_bars: int = 2, atr_period: int = 14,
                 atr_threshold: float = 0.8, volume_factor: float = 1.2,
                 use_volume: bool = True, trailing_stop_pct: float = 0.8,
                 trade: bool = False, trade_amount: float = 10.0,
                 trade_mode: int = 3, leverage: int = 3):
        """
        Initialize Strategy Monitor with unified mock/real trading logic
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            consecutive_bars: Number of consecutive bars for breakout
            atr_period: ATR calculation period
            atr_threshold: ATR threshold multiplier
            volume_factor: Volume expansion factor
            use_volume: Whether to use volume condition
            trailing_stop_pct: Trailing stop percentage
            trade: Whether to execute real trades (True) or mock trades (False)
            trade_amount: USDT amount for each trade
            trade_mode: Trading mode (1=现货, 2=全仓杠杆, 3=逐仓杠杆)
            leverage: Leverage multiplier (default: 3x)
        """
        self.trade_amount = trade_amount
        self.trade_mode_setting = trade_mode
        self.leverage = leverage

        self.consecutive_bars = consecutive_bars
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        self.volume_factor = volume_factor
        self.use_volume = use_volume
        self.trailing_stop_pct = trailing_stop_pct
        
        # 从环境变量获取API凭证（如果存在）
        api_key = os.getenv('OK-ACCESS-KEY')
        api_secret = os.getenv('OK-ACCESS-SECRET')
        passphrase = os.getenv('OK-ACCESS-PASSPHRASE')

        self.client = OKXClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase)
        self.strategy = HighFrequencyStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)

        # 先初始化trader，再初始化BaseMonitor
        self.trader = Strategy2Trader(self.client, trade_amount, trade_mode, leverage)
        
        from sdk.base_monitor import BaseMonitor
        BaseMonitor.__init__(self, symbol, bar, trade_mode=trade)
        
        # 预加载instrument信息到缓存
        self._preload_instrument_info()

    def _preload_instrument_info(self):
        """
        预加载instrument信息到缓存
        - 真实交易模式：如果加载失败则退出程序
        - 模拟交易模式：如果加载失败则发出警告
        """
        try:
            # 获取交易对的实际instrument ID（考虑杠杆模式）
            inst_id = self.trader.get_inst_id(self.symbol)
            
            logger.info(f"正在预加载instrument信息: {inst_id}")
            
            # 先检查JSON缓存
            from apis.okx_api.instrument_cache import InstrumentCache
            cache = InstrumentCache()
            cached_instrument = cache.get_instrument(inst_id)
            
            if cached_instrument:
                # 如果JSON缓存中有有效数据，直接使用
                self.trader._instrument_cache[inst_id] = cached_instrument
                logger.info(f"✅ 从JSON缓存加载instrument信息: {inst_id}")
                # Handle both Instrument object and dictionary
                if hasattr(cached_instrument, 'minSz'):
                    logger.info(f"   - 最小下单数量: {cached_instrument.minSz}")
                    logger.info(f"   - 合约面值: {cached_instrument.ctVal}")
                else:
                    logger.info(f"   - 最小下单数量: {cached_instrument.get('minSz', 'N/A')}")
                    logger.info(f"   - 合约面值: {cached_instrument.get('ctVal', 'N/A')}")
                return
            
            # 尝试计算订单大小来触发instrument信息获取和缓存
            # 先获取当前价格
            ticker = self.market_data_retriever.get_ticker_by_symbol(inst_id)
            if not ticker:
                if self.trade_mode:
                    logger.error(f"❌ 无法获取 {inst_id} 的价格信息，真实交易无法进行")
                    raise Exception(f"无法获取 {inst_id} 的价格信息")
                else:
                    logger.warning(f"⚠️ 无法获取 {inst_id} 的价格信息，模拟交易将继续运行")
                    return
            
            current_price = ticker.last
            
            # 调用calculate_order_size来触发instrument信息获取和缓存
            order_size = self.trader.calculate_order_size(inst_id, current_price)
            
            # 检查instrument信息是否成功缓存
            if inst_id in self.trader._instrument_cache:
                instrument = self.trader._instrument_cache[inst_id]
                logger.info(f"✅ 成功预加载instrument信息: {inst_id}")
                # Handle both Instrument object and dictionary
                if hasattr(instrument, 'minSz'):
                    logger.info(f"   - 最小下单数量: {instrument.minSz}")
                    logger.info(f"   - 合约价值: {instrument.ctVal}")
                else:
                    logger.info(f"   - 最小下单数量: {instrument.get('minSz', 'N/A')}")
                    logger.info(f"   - 合约价值: {instrument.get('ctVal', 'N/A')}")
                logger.info(f"   - 计算的下单数量: {order_size}")
            else:
                if self.trade_mode:
                    logger.error(f"❌ instrument信息未成功缓存，真实交易无法进行")
                    raise Exception(f"instrument信息未成功缓存")
                else:
                    logger.warning(f"⚠️  instrument信息未成功缓存，模拟交易将继续运行")
                
        except Exception as e:
            if self.trade_mode:
                logger.error(f"❌ 预加载instrument信息失败: {e}")
                logger.error(f"❌ 真实交易无法启动，原因可能是：")
                logger.error(f"   1. 网络连接问题 - 无法连接到OKX API")
                logger.error(f"   2. API配置错误 - 请检查API密钥和权限")
                logger.error(f"   3. 交易对不存在 - 请检查symbol是否正确")
                logger.error(f"❌ 请解决问题后重新启动监控进程")
                raise SystemExit("真实交易因instrument信息加载失败而退出")
            else:
                logger.warning(f"⚠️  预加载instrument信息失败: {e}")
                logger.warning(f"⚠️  模拟交易将继续运行，但下单数量计算可能不准确")

    def get_csv_headers(self) -> list:
        """Get CSV headers for recording data"""
        return [
            'timestamp', 'symbol', 'price', 'typical_price', 'prev_typical',
            'atr', 'atr_mean', 'atr_condition_met', 'volume_condition_met',
            'long_breakout', 'short_breakout', 'signal', 'action',
            'position', 'entry_price', 'exit_price', 'return_rate'
        ]

    def get_trader(self):
        """Get trader instance"""
        return self.trader

    def _check_trailing_stop(self, price: float) -> bool:
        """Check trailing stop condition for mock position"""
        if self.mock_position == 1:
            # 持多仓：更新最高价，检查是否跌破止损价
            if price > self.mock_highest_price:
                self.mock_highest_price = price
            stop_price = self.mock_highest_price * (1 - self.trailing_stop_pct / 100.0)
            if price <= stop_price:
                return True
        elif self.mock_position == -1:
            # 持空仓：更新最低价，检查是否涨破止损价
            if price < self.mock_lowest_price:
                self.mock_lowest_price = price
            stop_price = self.mock_lowest_price * (1 + self.trailing_stop_pct / 100.0)
            if price >= stop_price:
                return True
        return False

    def execute_trade(self, signal: int, price: float, details: dict):
        """Unified trading logic for both mock and real trading"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0

        trailing_stop_triggered = self._check_trailing_stop(price)

        if self.mock_position == 0:
            if signal == 1:
                self.mock_position = 1
                self.mock_entry_price = price
                self.mock_highest_price = price  # 持多仓时记录最高价
                self.mock_lowest_price = price  # 持多仓时也记录最低价用于参考
                action = "LONG_OPEN"
                self.trade_count += 1
            elif signal == -1:
                self.mock_position = -1
                self.mock_entry_price = price
                self.mock_lowest_price = price  # 持空仓时记录最低价
                self.mock_highest_price = price  # 持空仓时也记录最高价用于参考
                action = "SHORT_OPEN"
                self.trade_count += 1
        elif self.mock_position == 1:
            if trailing_stop_triggered:
                exit_price = price
                return_rate = (exit_price - self.mock_entry_price) / self.mock_entry_price
                action = "LONG_CLOSE_TRAILING_STOP"
                self.mock_position = 0
                self.mock_highest_price = 0.0
                self.trade_count += 1
            elif signal == -1:
                exit_price = price
                return_rate = (exit_price - self.mock_entry_price) / self.mock_entry_price
                action = "LONG_CLOSE_SHORT_OPEN"
                self.mock_position = -1
                self.mock_entry_price = price
                self.mock_highest_price = price  # 重新初始化
                self.mock_lowest_price = price  # 重新初始化
                self.trade_count += 1
        elif self.mock_position == -1:
            if trailing_stop_triggered:
                exit_price = price
                return_rate = (self.mock_entry_price - exit_price) / self.mock_entry_price
                action = "SHORT_CLOSE_TRAILING_STOP"
                self.mock_position = 0
                self.mock_lowest_price = 0.0
                self.trade_count += 1
            elif signal == 1:
                exit_price = price
                return_rate = (self.mock_entry_price - exit_price) / self.mock_entry_price
                action = "SHORT_CLOSE_LONG_OPEN"
                self.mock_position = 1
                self.mock_entry_price = price
                self.mock_lowest_price = price  # 重新初始化
                self.mock_highest_price = price  # 重新初始化
                self.trade_count += 1
        else:
            raise ValueError(f"mock position is invalid: {self.mock_position}")

        if action != "HOLD":
            # 记录模拟交易数据（始终记录）
            mock_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': self.symbol,
                'price': price,
                'typical_price': details.get('current_typical', 0),
                'prev_typical': details.get('prev_typical', 0),
                'atr': details.get('atr', 0),
                'atr_mean': details.get('atr_mean', 0),
                'atr_condition_met': details.get('atr_condition_met', False),
                'volume_condition_met': details.get('volume_condition_met', False),
                'long_breakout': details.get('long_breakout', False),
                'short_breakout': details.get('short_breakout', False),
                'signal': signal,
                'action': action,
                'position': self.mock_position,
                'entry_price': self.mock_entry_price,
                'exit_price': exit_price,
                'return_rate': return_rate
            }
            self._enqueue_mock_write(mock_record)

            # 如果是真实交易模式，执行真实交易并记录真实交易数据
            if self.trade_mode:
                # 执行真实交易
                trade_result = self.trader.execute_trade(action, self.symbol, price)

                # 记录真实交易数据
                if trade_result:
                    real_record = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': self.symbol,
                        'action': action,
                        'order_id': trade_result.ordId if hasattr(trade_result, 'ordId') else '',
                        'order_price': trade_result.px if hasattr(trade_result, 'px') else price,
                        'order_size': trade_result.sz if hasattr(trade_result, 'sz') else self.trade_amount,
                        'order_type': trade_result.ordType if hasattr(trade_result, 'ordType') else 'market',
                        'order_side': trade_result.side if hasattr(trade_result, 'side') else '',
                        'order_state': trade_result.state if hasattr(trade_result, 'state') else '',
                        'filled_size': trade_result.accFillSz if hasattr(trade_result, 'accFillSz') else 0,
                        'avg_fill_price': trade_result.avgPx if hasattr(trade_result, 'avgPx') else price,
                        'fee': trade_result.fee if hasattr(trade_result, 'fee') else 0,
                        'trade_timestamp': trade_result.ts if hasattr(trade_result, 'ts') else int(
                            datetime.now().timestamp() * 1000)
                    }
                    self._enqueue_real_write(real_record)

            mode_prefix = self.get_mode_prefix()
            logger.info(f"{mode_prefix} [{mock_record['timestamp']}] {self.symbol} {action}: "
                        f"价格={price:.4f}, 信号={signal}, 收益率={return_rate * 100:.2f}%")

    def run(self):
        """Run monitoring loop with unified mock/real trading logic"""
        mode_text = "真实交易" if self.trade_mode else "模拟监控"
        logger.info(f"开始{mode_text} {self.symbol} 的高频短线策略...")
        logger.info(f"策略参数: 连续突破={self.consecutive_bars}根K线, "
                    f"ATR周期={self.atr_period}, ATR阈值={self.atr_threshold}, "
                    f"成交量倍数={self.volume_factor}, 移动止损={self.trailing_stop_pct}%, "
                    f"使用成交量条件={'是' if self.use_volume else '否'}")

        if self.trade_mode:
            trade_mode_names = {1: "现货", 2: "全仓杠杆", 3: "逐仓杠杆"}
            logger.info(f"交易模式: {trade_mode_names.get(self.trade_mode_setting, '未知')}, "
                        f"每次交易金额: {self.trade_amount} USDT")
            if self.trade_mode_setting in [2, 3]:
                logger.info(f"杠杆倍数: {self.leverage}x")

        try:
            while True:
                self._wait_for_next_bar()

                try:
                    signal = self.strategy.calculate_high_frequency_signal(
                        self.symbol, self.bar, self.consecutive_bars,
                        self.atr_period, self.atr_threshold, self.volume_factor,
                        self.use_volume, self.mock_position
                    )

                    details = self.strategy.get_strategy_details(
                        self.symbol, self.bar, self.consecutive_bars,
                        self.atr_period, self.atr_threshold, self.volume_factor,
                        self.use_volume, self.mock_position
                    )

                    price = details.get('current_price', 0)

                    if price > 0:
                        # 在每次K线更新时都检查移动止损，确保最高价/最低价正确更新
                        self._check_trailing_stop(price)
                        self.execute_trade(signal, price, details)

                        # 计算距离最高价/最低价的百分比
                        price_info = ""
                        if self.mock_position == 1 and self.mock_highest_price > 0:
                            # 持多仓：计算当前价格距离最高价的百分比
                            distance_pct = (self.mock_highest_price - price) / self.mock_highest_price * 100
                            price_info = f", 距离最高价: {distance_pct:.2f}%"
                        elif self.mock_position == -1 and self.mock_lowest_price > 0:
                            # 持空仓：计算当前价格距离最低价的百分比
                            distance_pct = (price - self.mock_lowest_price) / self.mock_lowest_price * 100
                            price_info = f", 距离最低价: {distance_pct:.2f}%"

                        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"{self.symbol} 价格: {price:.4f}, 信号: {signal}, "
                                    f"模拟仓位: {self.mock_position}, 交易次数: {self.trade_count}{price_info}")
                    else:
                        logger.warning(f"无法获取 {self.symbol} 的价格数据")

                except Exception as e:
                    logger.error(f"计算信号时出错: {e}")
                    import time
                    time.sleep(5)

        except KeyboardInterrupt:
            logger.info("监控已停止")
            logger.info(f"总交易次数: {self.trade_count}")
            logger.info(f"模拟记录文件: {self.mock_csv_file}")
            logger.info(f"模拟备份文件: {self.mock_backup_file}")
            if self.trade_mode:
                logger.info(f"真实记录文件: {self.real_csv_file}")
                logger.info(f"真实备份文件: {self.real_backup_file}")
        except Exception as e:
            logger.error(f"监控过程中出错: {e}")


def load_config_from_file(config_path: str) -> dict:
    """从配置文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"从配置文件 {config_path} 加载配置成功")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def get_user_input(default_config: dict = None) -> dict:
    """获取用户输入参数，使用配置文件的默认值"""
    logger.info("高频短线策略实盘模拟监控系统")
    logger.info("=" * 50)

    # 使用配置文件的默认值
    if default_config is None:
        default_config = {}

    default_symbol = default_config.get('symbol', 'BTC-USDT')
    default_bar = default_config.get('bar', '1m')
    default_consecutive_bars = default_config.get('consecutive_bars', 2)
    default_atr_period = default_config.get('atr_period', 14)
    default_atr_threshold = default_config.get('atr_threshold', 0.8)
    default_volume_factor = default_config.get('volume_factor', 1.2)
    default_use_volume = default_config.get('use_volume', True)
    default_trailing_stop_pct = default_config.get('trailing_stop_pct', 0.8)
    default_trade = default_config.get('trade', False)
    default_trade_amount = default_config.get('trade_amount', 10.0)
    default_trade_mode = default_config.get('trade_mode', 3)
    default_leverage = default_config.get('leverage', 3)

    try:
        symbol = input(f"请输入交易对 (默认 {default_symbol}): ").strip() or default_symbol
        bar = input(f"请输入K线周期 (默认 {default_bar}): ").strip() or default_bar
        
        consecutive_bars_input = input(f"请输入连续突破K线数量 (默认 {default_consecutive_bars}): ").strip()
        consecutive_bars = int(consecutive_bars_input) if consecutive_bars_input else default_consecutive_bars
        
        atr_period_input = input(f"请输入ATR周期 (默认 {default_atr_period}): ").strip()
        atr_period = int(atr_period_input) if atr_period_input else default_atr_period
        
        atr_threshold_input = input(f"请输入ATR阈值 (默认 {default_atr_threshold}): ").strip()
        atr_threshold = float(atr_threshold_input) if atr_threshold_input else default_atr_threshold
        
        volume_factor_input = input(f"请输入成交量放大倍数 (默认 {default_volume_factor}): ").strip()
        volume_factor = float(volume_factor_input) if volume_factor_input else default_volume_factor
        
        use_volume_input = input(f"是否使用成交量条件 (y/n, 默认 {'y' if default_use_volume else 'n'}): ").strip().lower()
        if use_volume_input:
            use_volume = use_volume_input == 'y' or use_volume_input == 'yes'
        else:
            use_volume = default_use_volume
        
        trailing_stop_input = input(f"请输入移动止损百分比 (默认 {default_trailing_stop_pct}%): ").strip()
        trailing_stop_pct = float(trailing_stop_input) if trailing_stop_input else default_trailing_stop_pct

        trade_input = input(f"是否真实交易 (y/n, 默认 {'y' if default_trade else 'n'}): ").strip().lower()
        if trade_input:
            trade = trade_input == 'y' or trade_input == 'yes'
        else:
            trade = default_trade

        trade_amount = default_trade_amount
        trade_mode = default_trade_mode
        leverage = default_leverage

        if trade:
            trade_amount_input = input(f"请输入每次交易的USDT金额 (默认 {default_trade_amount}): ").strip()
            trade_amount = float(trade_amount_input) if trade_amount_input else default_trade_amount

            logger.info("交易模式选择:")
            logger.info("1. 现货模式")
            logger.info("2. 全仓杠杆模式")
            logger.info("3. 逐仓杠杆模式")

            trade_mode_names = {1: "现货", 2: "全仓杠杆", 3: "逐仓杠杆"}
            default_trade_mode_name = trade_mode_names.get(default_trade_mode, "逐仓杠杆")

            trade_mode_input = input(f"请选择交易模式 (1/2/3, 默认 {default_trade_mode}): ").strip()
            trade_mode = int(trade_mode_input) if trade_mode_input in ['1', '2', '3'] else default_trade_mode

            if trade_mode in [2, 3]:
                leverage_input = input(f"请输入杠杆倍数 (默认 {default_leverage}): ").strip()
                leverage = int(leverage_input) if leverage_input else default_leverage

    except (EOFError, KeyboardInterrupt):
        logger.info("\n用户取消输入，使用默认配置")
        # 使用默认配置
        symbol = default_symbol
        bar = default_bar
        consecutive_bars = default_consecutive_bars
        atr_period = default_atr_period
        atr_threshold = default_atr_threshold
        volume_factor = default_volume_factor
        use_volume = default_use_volume
        trailing_stop_pct = default_trailing_stop_pct
        trade = default_trade
        trade_amount = default_trade_amount
        trade_mode = default_trade_mode
        leverage = default_leverage

    return {
        'symbol': symbol,
        'bar': bar,
        'consecutive_bars': consecutive_bars,
        'atr_period': atr_period,
        'atr_threshold': atr_threshold,
        'volume_factor': volume_factor,
        'use_volume': use_volume,
        'trailing_stop_pct': trailing_stop_pct,
        'trade': trade,
        'trade_amount': trade_amount,
        'trade_mode': trade_mode,
        'leverage': leverage
    }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='高频短线策略实盘模拟监控系统')
    parser.add_argument('--config', type=str, help='配置文件路径', default=None)
    args = parser.parse_args()

    # 加载配置
    if args.config:
        # 加载用户指定的配置文件作为默认值
        default_config = load_config_from_file(args.config)
        if not default_config:
            logger.error("配置文件加载失败，使用系统默认值")
            default_config = {}
    else:
        # 加载默认配置文件作为用户输入的默认值
        config_path = os.path.join(os.path.dirname(__file__), 'configs/bnb_usdt_swap.json')
        default_config = load_config_from_file(config_path)
        if not default_config:
            logger.info("未找到默认配置文件，使用系统默认值")
            default_config = {}
    
    # 始终咨询用户输入，使用配置文件作为默认值
    config = get_user_input(default_config)

    # 设置默认值
    symbol = config.get('symbol', 'BTC-USDT')
    bar = config.get('bar', '1m')
    consecutive_bars = config.get('consecutive_bars', 2)
    atr_period = config.get('atr_period', 14)
    atr_threshold = config.get('atr_threshold', 0.8)
    volume_factor = config.get('volume_factor', 1.2)
    use_volume = config.get('use_volume', True)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.8)
    trade = config.get('trade', False)
    trade_amount = config.get('trade_amount', 10.0)
    trade_mode = config.get('trade_mode', 3)
    leverage = config.get('leverage', 3)

    # 打印最终使用的参数
    logger.info("\n" + "=" * 50)
    logger.info("本次运行使用的最终参数:")
    logger.info(f"  交易对: {symbol}")
    logger.info(f"  K线周期: {bar}")
    logger.info(f"  连续突破K线: {consecutive_bars}根")
    logger.info(f"  ATR周期: {atr_period}")
    logger.info(f"  ATR阈值: {atr_threshold}")
    logger.info(f"  成交量倍数: {volume_factor}")
    logger.info(f"  使用成交量条件: {'是' if use_volume else '否'}")
    logger.info(f"  移动止损: {trailing_stop_pct}%")
    logger.info(f"  真实交易: {'是' if trade else '否'}")
    
    if trade:
        trade_mode_names = {1: "现货", 2: "全仓杠杆", 3: "逐仓杠杆"}
        logger.info(f"  交易模式: {trade_mode_names.get(trade_mode, '未知')}")
        logger.info(f"  每次交易金额: {trade_amount} USDT")
        if trade_mode in [2, 3]:
            logger.info(f"  杠杆倍数: {leverage}x")
    
    logger.info("=" * 50 + "\n")

    # 创建监控器
    monitor = StrategyMonitor(
        symbol=symbol,
        bar=bar,
        consecutive_bars=consecutive_bars,
        atr_period=atr_period,
        atr_threshold=atr_threshold,
        volume_factor=volume_factor,
        use_volume=use_volume,
        trailing_stop_pct=trailing_stop_pct,
        trade=trade,
        trade_amount=trade_amount,
        trade_mode=trade_mode,
        leverage=leverage
    )

    monitor.run()


if __name__ == "__main__":
    main()