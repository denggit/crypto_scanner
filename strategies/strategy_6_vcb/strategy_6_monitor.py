#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2025
@File       : strategy_6_monitor.py
@Description: VCB策略实盘模拟监控系统，单币种
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

from strategies.strategy_6_vcb.methods.monitor import Strategy6Monitor
from strategies.strategy_6_vcb.methods.trader import Strategy6Trader
from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_6_vcb.strategy_6 import VCBStrategy
from utils.logger import logger


class StrategyMonitor(Strategy6Monitor):
    """Strategy Monitor with Unified Logic for Mock and Real Trading"""

    def __init__(self, symbol: str, bar: str = '1m',
                 atr_short_period: int = 10, atr_mid_period: int = 60,
                 atr_ratio_threshold: float = 0.5,
                 bb_period: int = 20, bb_std: int = 2,
                 bb_width_ratio: float = 0.7,
                 volume_period: int = 20, volume_multiplier: float = 1.0,
                 ttl_bars: int = 30,
                 trailing_stop_pct: float = 1.0,
                 stop_loss_atr_multiplier: float = 0.8,
                 take_profit_r: float = 2.0,
                 trade: bool = False, trade_amount: float = 10.0,
                 trade_mode: int = 3, leverage: int = 3, **params):
        """
        Initialize Strategy Monitor with unified mock/real trading logic
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            atr_short_period: 短期ATR周期
            atr_mid_period: 中期ATR周期
            atr_ratio_threshold: ATR比率阈值
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            bb_width_ratio: 布林带宽度收缩比率
            volume_period: 成交量均线周期
            volume_multiplier: 成交量放大倍数
            ttl_bars: 压缩事件TTL（K线数量）
            trailing_stop_pct: 移动止损百分比
            stop_loss_atr_multiplier: 止损ATR倍数
            take_profit_r: 止盈R倍数
            trade: Whether to execute real trades (True) or mock trades (False)
            trade_amount: USDT amount for each trade
            trade_mode: Trading mode (1=现货, 2=全仓杠杆, 3=逐仓杠杆)
            leverage: Leverage multiplier (default: 3x)
            **params: Additional parameters
        """
        self.trade_amount = trade_amount
        self.trade_mode_setting = trade_mode
        self.leverage = leverage

        self.atr_short_period = atr_short_period
        self.atr_mid_period = atr_mid_period
        self.atr_ratio_threshold = atr_ratio_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_width_ratio = bb_width_ratio
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.ttl_bars = ttl_bars
        self.trailing_stop_pct = trailing_stop_pct
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_r = take_profit_r

        # 从环境变量获取API凭证（如果存在）
        api_key = os.getenv('OK-ACCESS-KEY')
        api_secret = os.getenv('OK-ACCESS-SECRET')
        passphrase = os.getenv('OK-ACCESS-PASSPHRASE')

        self.client = OKXClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase)
        self.strategy = VCBStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)

        # 先初始化trader，再初始化BaseMonitor
        self.trader = Strategy6Trader(self.client, trade_amount, trade_mode, leverage)
        
        from sdk.base_monitor import BaseMonitor
        BaseMonitor.__init__(self, symbol, bar, trade_mode=trade)
        
        # 初始化止损和止盈价格
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        
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

    def get_trader(self):
        """Get trader instance"""
        return self.trader

    def execute_trade(self, signal: int, price: float, details: dict):
        """Unified trading logic for both mock and real trading"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0

        trailing_stop_triggered = self._check_trailing_stop(price)
        stop_loss_triggered, stop_type = self._check_stop_loss_take_profit(price)

        if self.mock_position == 0:
            if signal == 1:
                self.mock_position = 1
                self.mock_entry_price = price
                self.mock_highest_price = price
                self.mock_lowest_price = price
                
                # 计算止损和止盈
                compression_event = None
                if self.symbol in self.strategy.compression_pool:
                    compression_event = self.strategy.compression_pool[self.symbol]
                
                self.stop_loss_price, self.take_profit_price = self._calculate_stop_loss_take_profit(
                    price, 1, compression_event
                )
                
                action = "LONG_OPEN"
                self.trade_count += 1
            elif signal == -1:
                self.mock_position = -1
                self.mock_entry_price = price
                self.mock_lowest_price = price
                self.mock_highest_price = price
                
                # 计算止损和止盈
                compression_event = None
                if self.symbol in self.strategy.compression_pool:
                    compression_event = self.strategy.compression_pool[self.symbol]
                
                self.stop_loss_price, self.take_profit_price = self._calculate_stop_loss_take_profit(
                    price, -1, compression_event
                )
                
                action = "SHORT_OPEN"
                self.trade_count += 1
        elif self.mock_position == 1:
            if stop_loss_triggered:
                exit_price = price
                return_rate = (exit_price - self.mock_entry_price) / self.mock_entry_price
                action = f"LONG_CLOSE_{stop_type}"
                self.mock_position = 0
                self.mock_highest_price = 0.0
                self.stop_loss_price = 0.0
                self.take_profit_price = 0.0
                self.trade_count += 1
            elif trailing_stop_triggered:
                exit_price = price
                return_rate = (exit_price - self.mock_entry_price) / self.mock_entry_price
                action = "LONG_CLOSE_TRAILING_STOP"
                self.mock_position = 0
                self.mock_highest_price = 0.0
                self.stop_loss_price = 0.0
                self.take_profit_price = 0.0
                self.trade_count += 1
            elif signal == -1:
                exit_price = price
                return_rate = (exit_price - self.mock_entry_price) / self.mock_entry_price
                action = "LONG_CLOSE_SHORT_OPEN"
                self.mock_position = -1
                self.mock_entry_price = price
                self.mock_highest_price = price
                self.mock_lowest_price = price
                
                # 重新计算止损和止盈
                compression_event = None
                if self.symbol in self.strategy.compression_pool:
                    compression_event = self.strategy.compression_pool[self.symbol]
                
                self.stop_loss_price, self.take_profit_price = self._calculate_stop_loss_take_profit(
                    price, -1, compression_event
                )
                self.trade_count += 1
        elif self.mock_position == -1:
            if stop_loss_triggered:
                exit_price = price
                return_rate = (self.mock_entry_price - exit_price) / self.mock_entry_price
                action = f"SHORT_CLOSE_{stop_type}"
                self.mock_position = 0
                self.mock_lowest_price = 0.0
                self.stop_loss_price = 0.0
                self.take_profit_price = 0.0
                self.trade_count += 1
            elif trailing_stop_triggered:
                exit_price = price
                return_rate = (self.mock_entry_price - exit_price) / self.mock_entry_price
                action = "SHORT_CLOSE_TRAILING_STOP"
                self.mock_position = 0
                self.mock_lowest_price = 0.0
                self.stop_loss_price = 0.0
                self.take_profit_price = 0.0
                self.trade_count += 1
            elif signal == 1:
                exit_price = price
                return_rate = (self.mock_entry_price - exit_price) / self.mock_entry_price
                action = "SHORT_CLOSE_LONG_OPEN"
                self.mock_position = 1
                self.mock_entry_price = price
                self.mock_lowest_price = price
                self.mock_highest_price = price
                
                # 重新计算止损和止盈
                compression_event = None
                if self.symbol in self.strategy.compression_pool:
                    compression_event = self.strategy.compression_pool[self.symbol]
                
                self.stop_loss_price, self.take_profit_price = self._calculate_stop_loss_take_profit(
                    price, 1, compression_event
                )
                self.trade_count += 1
        else:
            raise ValueError(f"mock position is invalid: {self.mock_position}")

        if action != "HOLD":
            # 记录模拟交易数据（始终记录）
            mock_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': self.symbol,
                'price': price,
                'atr_ratio': details.get('atr_ratio', 0),
                'bb_width': details.get('bb_width', 0),
                'bb_upper': details.get('bb_upper', 0),
                'bb_lower': details.get('bb_lower', 0),
                'volume_expansion': details.get('volume_expansion', False),
                'signal': signal,
                'action': action,
                'position': self.mock_position,
                'entry_price': self.mock_entry_price,
                'exit_price': exit_price,
                'return_rate': return_rate,
                'stop_loss': self.stop_loss_price,
                'take_profit': self.take_profit_price
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
        logger.info(f"开始{mode_text} {self.symbol} 的VCB策略...")
        logger.info(f"策略参数: ATR({self.atr_short_period}/{self.atr_mid_period}), "
                    f"ATR比率阈值={self.atr_ratio_threshold}, "
                    f"BB({self.bb_period}, {self.bb_std}), "
                    f"成交量倍数={self.volume_multiplier}, "
                    f"TTL={self.ttl_bars}根K线, "
                    f"移动止损={self.trailing_stop_pct}%, "
                    f"止损ATR倍数={self.stop_loss_atr_multiplier}, "
                    f"止盈R倍数={self.take_profit_r}")

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
                    # 清理压缩池
                    self.strategy.cleanup_compression_pool(
                        symbol=self.symbol,
                        bar=self.bar,
                        atr_short_period=self.atr_short_period,
                        atr_mid_period=self.atr_mid_period
                    )

                    # 检测压缩
                    compression = self.strategy.detect_compression(
                        symbol=self.symbol,
                        bar=self.bar,
                        atr_short_period=self.atr_short_period,
                        atr_mid_period=self.atr_mid_period,
                        atr_ratio_threshold=self.atr_ratio_threshold,
                        bb_period=self.bb_period,
                        bb_std=self.bb_std,
                        bb_width_ratio=self.bb_width_ratio,
                        ttl_bars=self.ttl_bars
                    )

                    # 检测突破
                    signal, details = self.strategy.detect_breakout(
                        symbol=self.symbol,
                        bar=self.bar,
                        volume_period=self.volume_period,
                        volume_multiplier=self.volume_multiplier
                    )

                    price = details.get('current_price', 0)

                    if price > 0:
                        # 在每次K线更新时都检查移动止损和止损止盈
                        self._check_trailing_stop(price)
                        self.execute_trade(signal, price, details)

                        # 计算距离最高价/最低价的百分比
                        price_info = ""
                        if self.mock_position == 1 and self.mock_highest_price > 0:
                            distance_pct = (self.mock_highest_price - price) / self.mock_highest_price * 100
                            price_info = f", 距离最高价: {distance_pct:.2f}%"
                        elif self.mock_position == -1 and self.mock_lowest_price > 0:
                            distance_pct = (price - self.mock_lowest_price) / self.mock_lowest_price * 100
                            price_info = f", 距离最低价: {distance_pct:.2f}%"

                        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"{self.symbol} 价格: {price:.4f}, 信号: {signal}, "
                                    f"模拟仓位: {self.mock_position}, 交易次数: {self.trade_count}, "
                                    f"压缩池大小: {self.strategy.get_compression_pool_size()}{price_info}")
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
    from strategies.strategy_6_vcb.shared_config import get_user_input as shared_get_user_input
    return shared_get_user_input(default_config)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='VCB策略实盘模拟监控系统')
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
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.json')
        if os.path.exists(config_path):
            default_config = load_config_from_file(config_path)
            if not default_config:
                logger.info("未找到默认配置文件，使用系统默认值")
                default_config = {}
        else:
            logger.info("未找到默认配置文件，使用系统默认值")
            default_config = {}
    
    # 始终咨询用户输入，使用配置文件作为默认值
    config = get_user_input(default_config)

    # 设置默认值
    symbol = config.get('symbol', 'BTC-USDT')
    bar = config.get('bar', '1m')
    atr_short_period = config.get('atr_short_period', 10)
    atr_mid_period = config.get('atr_mid_period', 60)
    atr_ratio_threshold = config.get('atr_ratio_threshold', 0.5)
    bb_period = config.get('bb_period', 20)
    bb_std = config.get('bb_std', 2)
    bb_width_ratio = config.get('bb_width_ratio', 0.7)
    volume_period = config.get('volume_period', 20)
    volume_multiplier = config.get('volume_multiplier', 1.0)
    ttl_bars = config.get('ttl_bars', 30)
    trailing_stop_pct = config.get('trailing_stop_pct', 1.0)
    stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 0.8)
    take_profit_r = config.get('take_profit_r', 2.0)
    trade = config.get('trade', False)
    trade_amount = config.get('trade_amount', 10.0)
    trade_mode = config.get('trade_mode', 3)
    leverage = config.get('leverage', 3)

    # 打印最终使用的参数
    from strategies.strategy_6_vcb.shared_config import print_final_config
    print_final_config(config)

    # 创建监控器
    monitor = StrategyMonitor(
        symbol=symbol,
        bar=bar,
        atr_short_period=atr_short_period,
        atr_mid_period=atr_mid_period,
        atr_ratio_threshold=atr_ratio_threshold,
        bb_period=bb_period,
        bb_std=bb_std,
        bb_width_ratio=bb_width_ratio,
        volume_period=volume_period,
        volume_multiplier=volume_multiplier,
        ttl_bars=ttl_bars,
        trailing_stop_pct=trailing_stop_pct,
        stop_loss_atr_multiplier=stop_loss_atr_multiplier,
        take_profit_r=take_profit_r,
        trade=trade,
        trade_amount=trade_amount,
        trade_mode=trade_mode,
        leverage=leverage
    )

    monitor.run()


if __name__ == "__main__":
    main()

