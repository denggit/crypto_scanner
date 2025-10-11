#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/10/2025
@File       : strategy_1_monitor.py
@Description: EMA交叉策略实盘模拟监控系统
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

from strategies.strategy_1.methods.monitor import Strategy1Monitor
from strategies.strategy_1.methods.trader import Strategy1Trader
from strategies.strategy_1.methods.volatility_exit import VolatilityExit
from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_1.strategy_1 import EMACrossoverStrategy
from utils.logger import logger


class StrategyMonitor(Strategy1Monitor):
    """Strategy Monitor with Unified Logic for Mock and Real Trading"""

    def __init__(self, symbol: str, bar: str = '1m',
                 short_ma: int = 5, long_ma: int = 20,
                 mode: str = 'strict', trailing_stop_pct: float = 1.0,
                 trade: bool = False, trade_amount: float = 10.0,
                 trade_mode: int = 3, leverage: int = 3, assist_cond: str = 'volume',
                 volatility_exit: bool = False, volatility_threshold: float = 0.5, **params):
        """
        Initialize Strategy Monitor with unified mock/real trading logic
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            short_ma: Short EMA period
            long_ma: Long EMA period
            mode: Mode ('strict' or 'loose')
            trailing_stop_pct: Trailing stop percentage
            trade: Whether to execute real trades (True) or mock trades (False)
            trade_amount: USDT amount for each trade
            trade_mode: Trading mode (1=现货, 2=全仓杠杆, 3=逐仓杠杆)
            leverage: Leverage multiplier (default: 3x)
            assist_cond: Assist condition type ('volume', 'rsi', or None, default: 'volume')
            volatility_exit: Whether to enable volatility-based exit
            volatility_threshold: Volatility threshold for exit (0.5 means 50% reduction)
            **params: Additional parameters for assist conditions
                  vol_multiplier: Volume multiplier (default: 1.2)
                  confirmation_pct: Confirmation percentage (default: 0.2)
                  rsi_period: RSI period (default: 9)
        """
        self.trade_amount = trade_amount
        self.trade_mode_setting = trade_mode
        self.leverage = leverage

        self.short_ma = short_ma
        self.long_ma = long_ma
        self.mode = mode
        self.trailing_stop_pct = trailing_stop_pct
        self.assist_cond = assist_cond
        
        # 波动率退出参数
        self.volatility_exit = volatility_exit
        self.volatility_threshold = volatility_threshold
        
        # 设置默认参数并合并用户提供的参数
        self.params = {
            'vol_multiplier': 1.2,
            'confirmation_pct': 0.2
        }
        self.params.update(params)

        # 从环境变量获取API凭证（如果存在）
        api_key = os.getenv('OK-ACCESS-KEY')
        api_secret = os.getenv('OK-ACCESS-SECRET')
        passphrase = os.getenv('OK-ACCESS-PASSPHRASE')

        self.client = OKXClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase)
        self.strategy = EMACrossoverStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)

        # 先初始化trader，再初始化BaseMonitor
        self.trader = Strategy1Trader(self.client, trade_amount, trade_mode, leverage)
        
        from sdk.base_monitor import BaseMonitor
        BaseMonitor.__init__(self, symbol, bar, trade_mode=trade)
        
        # 初始化波动率退出条件
        self.volatility_exit_checker = VolatilityExit(short_ma, volatility_threshold)
        
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
            inst_id = self.trader._get_inst_id(self.symbol)
            
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
            'timestamp', 'symbol', 'price', 'ema5', 'ema20', 'ema20_slope',
            'volume_expansion', 'volume_ratio', 'signal', 'action',
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

    def _check_volatility_exit(self) -> bool:
        """检查波动率退出条件"""
        if not self.volatility_exit or self.mock_position == 0:
            return False
        
        return self.volatility_exit_checker.check_volatility_exit(self.mock_position)

    def execute_trade(self, signal: int, price: float, details: dict):
        """Unified trading logic for both mock and real trading"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0

        # 更新波动率退出检查器的价格历史记录
        self.volatility_exit_checker.update_price_history(price)

        trailing_stop_triggered = self._check_trailing_stop(price)
        volatility_exit_triggered = self._check_volatility_exit()

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
            elif volatility_exit_triggered:
                exit_price = price
                return_rate = (exit_price - self.mock_entry_price) / self.mock_entry_price
                action = "LONG_CLOSE_VOLATILITY"
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
            elif volatility_exit_triggered:
                exit_price = price
                return_rate = (self.mock_entry_price - exit_price) / self.mock_entry_price
                action = "SHORT_CLOSE_VOLATILITY"
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
                'ema5': details.get('ema5', 0),
                'ema20': details.get('ema20', 0),
                'ema20_slope': details.get('ema20_slope', 0),
                'volume_expansion': details.get('volume_expansion', False),
                'volume_ratio': details.get('volume_ratio', 0),
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
        logger.info(f"开始{mode_text} {self.symbol} 的EMA交叉策略...")
        logger.info(f"策略参数: EMA{self.short_ma}/EMA{self.long_ma}, "
                    f"成交量倍数={self.params.get('vol_multiplier', 1.2)}, 确认百分比={self.params.get('confirmation_pct', 0.2)}%, "
                    f"模式={self.mode}, 移动止损={self.trailing_stop_pct}%, "
                    f"辅助条件={self.assist_cond if self.assist_cond else '无'}, "
                    f"波动率退出={'是' if self.volatility_exit else '否'}, 阈值={self.volatility_threshold}")

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
                    signal = self.strategy.calculate_ema_crossover_signal(
                        self.symbol, self.bar, self.short_ma, self.long_ma,
                        self.mode, self.assist_cond, **self.params
                    )

                    details = self.strategy.get_strategy_details(
                        self.symbol, self.bar, self.short_ma, self.long_ma,
                        self.mode, self.assist_cond, **self.params
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
    logger.info("EMA交叉策略实盘模拟监控系统")
    logger.info("=" * 50)

    # 使用配置文件的默认值
    if default_config is None:
        default_config = {}

    default_symbol = default_config.get('symbol', 'BTC-USDT')
    default_bar = default_config.get('bar', '1m')
    default_mode = default_config.get('mode', 'loose')
    default_trailing_stop_pct = default_config.get('trailing_stop_pct', 1.0)
    default_trade = default_config.get('trade', False)
    default_trade_amount = default_config.get('trade_amount', 10.0)
    default_trade_mode = default_config.get('trade_mode', 3)
    default_leverage = default_config.get('leverage', 3)
    default_assist_cond = default_config.get('assist_cond', 'volume')
    default_volatility_exit = default_config.get('volatility_exit', False)
    default_volatility_threshold = default_config.get('volatility_threshold', 0.5)
    default_params = default_config.get('params', {})

    try:
        symbol = input(f"请输入交易对 (默认 {default_symbol}): ").strip() or default_symbol
        bar = input(f"请输入K线周期 (默认 {default_bar}): ").strip() or default_bar
        
        short_ma_input = input(f"请输入短EMA周期 (默认 {default_config.get('short_ma', 5)}): ").strip()
        short_ma = int(short_ma_input) if short_ma_input else default_config.get('short_ma', 5)
        
        long_ma_input = input(f"请输入长EMA周期 (默认 {default_config.get('long_ma', 20)}): ").strip()
        long_ma = int(long_ma_input) if long_ma_input else default_config.get('long_ma', 20)
        
        mode = input(f"请输入模式 (strict/loose, 默认 {default_mode}): ").strip() or default_mode
        trailing_stop_input = input(f"请输入移动止损百分比 (默认 {default_trailing_stop_pct}%): ").strip()
        trailing_stop_pct = float(trailing_stop_input) if trailing_stop_input else default_trailing_stop_pct

        trade_input = input(f"是否真实交易 (y/n, 默认 {'y' if default_trade else 'n'}): ").strip().lower()
        if trade_input:
            trade = trade_input == 'y' or trade_input == 'yes'
        else:
            trade = default_trade

        # 波动率退出条件
        volatility_exit_input = input(f"是否启用波动率退出条件 (y/n, 默认 {'y' if default_volatility_exit else 'n'}): ").strip().lower()
        if volatility_exit_input:
            volatility_exit = volatility_exit_input == 'y' or volatility_exit_input == 'yes'
        else:
            volatility_exit = default_volatility_exit

        volatility_threshold = default_volatility_threshold
        if volatility_exit:
            volatility_threshold_input = input(f"请输入波动率阈值 (默认 {default_volatility_threshold}): ").strip()
            volatility_threshold = float(volatility_threshold_input) if volatility_threshold_input else default_volatility_threshold

        logger.info("辅助条件选择:")
        logger.info("1. 成交量放大 (volume)")
        logger.info("2. RSI条件 (rsi)")
        logger.info("3. 无辅助条件 (none)")

        assist_cond_map = {'1': 'volume', '2': 'rsi', '3': None}
        default_assist_input = '1' if default_assist_cond == 'volume' else '2' if default_assist_cond == 'rsi' else '3'

        assist_input = input(f"请选择辅助条件 (1/2/3, 默认 {default_assist_input}): ").strip()
        assist_cond = assist_cond_map.get(assist_input, default_assist_cond)

        # 根据选择的辅助条件动态获取对应的技术指标参数
        params = default_params.copy()  # 复制默认参数
        
        if assist_cond == 'volume':
            # 成交量辅助条件：询问成交量相关参数
            default_vol_multiplier = default_params.get('vol_multiplier', 1.2)
            default_confirmation_pct = default_params.get('confirmation_pct', 0.2)
            
            vol_multiplier_input = input(f"请输入成交量放大倍数 (默认 {default_vol_multiplier}): ").strip()
            params['vol_multiplier'] = float(vol_multiplier_input) if vol_multiplier_input else default_vol_multiplier
            
            confirmation_pct_input = input(f"请输入确认突破百分比 (默认 {default_confirmation_pct}%): ").strip()
            params['confirmation_pct'] = float(confirmation_pct_input) if confirmation_pct_input else default_confirmation_pct
        
        elif assist_cond == 'rsi':
            # RSI辅助条件：询问RSI相关参数
            default_rsi_period = default_params.get('rsi_period', 9)
            rsi_period_input = input(f"请输入RSI周期 (默认 {default_rsi_period}): ").strip()
            params['rsi_period'] = int(rsi_period_input) if rsi_period_input else default_rsi_period

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
        short_ma = default_config.get('short_ma', 5)
        long_ma = default_config.get('long_ma', 20)
        mode = default_mode
        trailing_stop_pct = default_trailing_stop_pct
        trade = default_trade
        assist_cond = default_assist_cond
        volatility_exit = default_volatility_exit
        volatility_threshold = default_volatility_threshold
        params = default_params.copy()
        trade_amount = default_trade_amount
        trade_mode = default_trade_mode
        leverage = default_leverage

    return {
        'symbol': symbol,
        'bar': bar,
        'short_ma': short_ma,
        'long_ma': long_ma,
        'mode': mode,
        'trailing_stop_pct': trailing_stop_pct,
        'trade': trade,
        'trade_amount': trade_amount,
        'trade_mode': trade_mode,
        'leverage': leverage,
        'assist_cond': assist_cond,
        'volatility_exit': volatility_exit,
        'volatility_threshold': volatility_threshold,
        'params': params
    }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='EMA交叉策略实盘模拟监控系统')
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
    short_ma = config.get('short_ma', 5)
    long_ma = config.get('long_ma', 20)
    mode = config.get('mode', 'loose')
    trailing_stop_pct = config.get('trailing_stop_pct', 1.0)
    trade = config.get('trade', False)
    trade_amount = config.get('trade_amount', 10.0)
    trade_mode = config.get('trade_mode', 3)
    leverage = config.get('leverage', 3)
    assist_cond = config.get('assist_cond', 'volume')
    volatility_exit = config.get('volatility_exit', False)
    volatility_threshold = config.get('volatility_threshold', 0.5)
    params = config.get('params', {})

    # 打印最终使用的参数
    logger.info("\n" + "=" * 50)
    logger.info("本次运行使用的最终参数:")
    logger.info(f"  交易对: {symbol}")
    logger.info(f"  K线周期: {bar}")
    logger.info(f"  EMA参数: {short_ma}/{long_ma}")
    logger.info(f"  模式: {mode}")
    logger.info(f"  移动止损: {trailing_stop_pct}%")
    logger.info(f"  真实交易: {'是' if trade else '否'}")
    logger.info(f"  辅助条件: {assist_cond if assist_cond else '无'}")
    logger.info(f"  波动率退出: {'是' if volatility_exit else '否'}")
    if volatility_exit:
        logger.info(f"  波动率阈值: {volatility_threshold}")
    
    if assist_cond == 'volume':
        logger.info(f"  成交量倍数: {params.get('vol_multiplier', 1.2)}")
        logger.info(f"  确认百分比: {params.get('confirmation_pct', 0.2)}%")
    elif assist_cond == 'rsi':
        logger.info(f"  RSI周期: {params.get('rsi_period', 9)}")
    
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
        short_ma=short_ma,
        long_ma=long_ma,
        mode=mode,
        trailing_stop_pct=trailing_stop_pct,
        trade=trade,
        trade_amount=trade_amount,
        trade_mode=trade_mode,
        leverage=leverage,
        assist_cond=assist_cond,
        volatility_exit=volatility_exit,
        volatility_threshold=volatility_threshold,
        **params
    )

    monitor.run()


if __name__ == "__main__":
    main()
