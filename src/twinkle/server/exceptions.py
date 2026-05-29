"""Twinkle Server unified exception hierarchy."""

from __future__ import annotations


class TwinkleServerError(Exception):
    """所有 Twinkle Server 异常的基类"""
    pass


class StateBackendError(TwinkleServerError):
    """状态后端操作失败（连接断开、超时、数据序列化错误等）"""
    pass


class ConfigMismatchError(TwinkleServerError):
    """配置签名不匹配 — 重启后检测到配置变更，持久化数据可能与当前配置不兼容"""
    pass


class ResourceExhaustedError(TwinkleServerError):
    """资源耗尽 — 队列满、内存不足、连接池耗尽等"""
    pass
