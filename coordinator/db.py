from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from coordinator.models import Base


def build_engine():
    url = os.environ.get("DATABASE_URL", "sqlite:///./data/fedroom.db")
    os.makedirs(os.path.dirname(url.split("///")[-1]) or ".", exist_ok=True) if url.startswith("sqlite") else None
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return engine


def build_sessionmaker():
    engine = build_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)
