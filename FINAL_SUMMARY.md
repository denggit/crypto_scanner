# Crypto Scanner 项目重构完成

## 项目结构调整说明

我已经按照您的要求重新组织了项目结构：

### 1. okx_api/ 目录
只包含与OKX交易所API直接相关的代码：
- `client.py`: 主API客户端实现
- `market_data.py`: 市场数据接口
- `trader.py`: 交易接口
- `models.py`: 数据模型定义
- 示例代码和文档

### 2. tools/ 目录
包含各种工具类应用：
- `market_scanner.py`: 市场数据扫描和分析工具

### 3. utils/ 目录
包含实用工具和辅助脚本：
- `test_okx_connectivity.py`: 连接性测试工具

## 主要改进

1. **正确的项目架构**：按照功能分离代码，OKX API相关代码放在`okx_api/`目录，工具类代码放在`tools/`和`utils/`目录

2. **清晰的目录结构**：
   - `okx_api/`: 核心API模块
   - `tools/`: 应用工具
   - `utils/`: 实用工具

3. **便捷的使用方式**：
   - 提供了`run_scanner.py`启动脚本
   - 更新了README文档说明新的使用方法

## 使用方法

1. 安装依赖：
   ```bash
   pip install -r okx_api/requirements.txt
   ```

2. 配置API密钥到`.env`文件

3. 运行市场扫描工具：
   ```bash
   python run_scanner.py
   ```
   或
   ```bash
   python tools/market_scanner.py
   ```

4. 测试连接性：
   ```bash
   python utils/test_okx_connectivity.py
   ```

## 注意事项

网络连接问题可能是由于：
1. 网络连接不稳定
2. 防火墙或代理设置
3. 地区访问限制
4. OKX API临时不可用

请检查您的网络设置后再试。