"""
Root conftest.py — sets required environment variables before any test module
imports are collected, so pydantic-settings doesn't raise a ValidationError.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
