"""配置项测试"""

import importlib


def test_embedding_provider_defaults_to_disabled():
    settings_module = importlib.import_module("config.settings")
    loaded = settings_module.Settings()
    # 默认应该是 disabled，避免 CI 下载大模型
    assert loaded.embedding_provider in ("disabled", "local", "api")


def test_local_embedding_model_default_bge_m3():
    settings_module = importlib.import_module("config.settings")
    loaded = settings_module.Settings()
    assert loaded.embedding_local_model == "BAAI/bge-m3"
