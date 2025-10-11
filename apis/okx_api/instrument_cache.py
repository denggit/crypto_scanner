#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : instrument_cache.py
@Description: Instrument information cache management
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from utils.logger import logger


class InstrumentCache:
    """
    Instrument information cache manager
    Handles reading and writing instrument data to local JSON file
    """
    
    def __init__(self, cache_file_path: str = None):
        """
        Initialize Instrument Cache
        
        Args:
            cache_file_path: Path to the cache JSON file
        """
        if cache_file_path is None:
            # Default cache file location
            cache_dir = os.path.dirname(os.path.abspath(__file__))
            self.cache_file = os.path.join(cache_dir, "instrument_cache.json")
        else:
            self.cache_file = cache_file_path
        
        # Ensure cache directory exists
        cache_dir = os.path.dirname(self.cache_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Load existing cache
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache data from JSON file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"✅ 加载instrument缓存文件: {self.cache_file}")
                    return data
            else:
                logger.info(f"📝 创建新的instrument缓存文件: {self.cache_file}")
                return {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "instruments": {}
                }
        except Exception as e:
            logger.error(f"❌ 加载instrument缓存失败: {e}")
            return {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "instruments": {}
            }
    
    def _save_cache(self):
        """Save cache data to JSON file"""
        try:
            self.cache_data["last_updated"] = datetime.now().isoformat()
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"💾 保存instrument缓存到文件: {self.cache_file}")
        except Exception as e:
            logger.error(f"❌ 保存instrument缓存失败: {e}")
    
    def get_instrument(self, inst_id: str) -> Optional[Dict[str, Any]]:
        """
        Get instrument information from cache
        
        Args:
            inst_id: Instrument ID
            
        Returns:
            Instrument data if found and not expired, None otherwise
        """
        try:
            instruments = self.cache_data.get("instruments", {})
            if inst_id in instruments:
                instrument_data = instruments[inst_id]
                
                # Check if cache is expired (24 hours)
                last_updated_str = instrument_data.get("last_updated")
                if last_updated_str:
                    last_updated = datetime.fromisoformat(last_updated_str)
                    if datetime.now() - last_updated < timedelta(hours=24):
                        logger.debug(f"📖 从缓存读取instrument信息: {inst_id}")
                        return instrument_data["data"]
                    else:
                        logger.info(f"⏰ instrument缓存已过期: {inst_id}")
                        return None
                else:
                    logger.debug(f"📖 从缓存读取instrument信息: {inst_id}")
                    return instrument_data["data"]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 读取instrument缓存失败: {e}")
            return None
    
    def save_instrument(self, inst_id: str, instrument_data: Dict[str, Any]):
        """
        Save instrument information to cache
        
        Args:
            inst_id: Instrument ID
            instrument_data: Instrument data to cache
        """
        try:
            if "instruments" not in self.cache_data:
                self.cache_data["instruments"] = {}
            
            # Convert Instrument object to serializable dictionary
            serializable_data = {}
            if hasattr(instrument_data, '__dict__'):
                # If it's an object, convert to dict
                serializable_data = instrument_data.__dict__.copy()
            elif isinstance(instrument_data, dict):
                # If it's already a dict, use it directly
                serializable_data = instrument_data.copy()
            else:
                # Try to convert to dict using vars()
                try:
                    serializable_data = vars(instrument_data).copy()
                except:
                    logger.error(f"❌ 无法序列化instrument数据: {type(instrument_data)}")
                    return
            
            # Remove any non-serializable attributes
            for key in list(serializable_data.keys()):
                if not isinstance(serializable_data[key], (str, int, float, bool, list, dict, type(None))):
                    serializable_data[key] = str(serializable_data[key])
            
            self.cache_data["instruments"][inst_id] = {
                "data": serializable_data,
                "last_updated": datetime.now().isoformat(),
                "inst_id": inst_id
            }
            
            self._save_cache()
            logger.info(f"💾 保存instrument信息到缓存: {inst_id}")
            
        except Exception as e:
            logger.error(f"❌ 保存instrument信息到缓存失败: {e}")
    
    def get_all_instruments(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all cached instruments
        
        Returns:
            Dictionary of all cached instruments
        """
        return self.cache_data.get("instruments", {})
    
    def clear_expired_instruments(self):
        """Clear expired instruments from cache"""
        try:
            instruments = self.cache_data.get("instruments", {})
            expired_count = 0
            
            for inst_id, instrument_data in list(instruments.items()):
                last_updated_str = instrument_data.get("last_updated")
                if last_updated_str:
                    last_updated = datetime.fromisoformat(last_updated_str)
                    if datetime.now() - last_updated >= timedelta(hours=24):
                        del instruments[inst_id]
                        expired_count += 1
            
            if expired_count > 0:
                self._save_cache()
                logger.info(f"🗑️  清理了 {expired_count} 个过期的instrument缓存")
                
        except Exception as e:
            logger.error(f"❌ 清理过期instrument缓存失败: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics and information
        
        Returns:
            Cache information dictionary
        """
        instruments = self.cache_data.get("instruments", {})
        return {
            "cache_file": self.cache_file,
            "total_instruments": len(instruments),
            "created_at": self.cache_data.get("created_at"),
            "last_updated": self.cache_data.get("last_updated"),
            "version": self.cache_data.get("version")
        }