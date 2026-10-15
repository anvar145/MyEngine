"""Фоновые задачи ядра: пул потоков с доставкой результатов в главный поток."""

from __future__ import annotations

from engine.core.jobs.job_system import Job, JobSystem

__all__ = ["Job", "JobSystem"]
