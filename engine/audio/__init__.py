"""Аудио движка: бэкенды, клипы, микшер (A9a).

Ядро (``engine.core``) от аудио не зависит; устройство вывода подключается
только в плеере (extra ``[audio]``).
"""

from __future__ import annotations

from engine.audio.backend import (
    BUSES,
    AudioBackendError,
    IAudioBackend,
    VoiceHandle,
)
from engine.audio.backends import NullBackend, SoftwareBackend, create_backend
from engine.audio.clip import (
    CLIP_CACHE_LIMIT,
    AudioClip,
    AudioDecodeError,
    ClipCache,
    decode_audio,
    decode_wav,
    load_audio_clip,
)
from engine.audio.components import AudioListenerComponent, AudioSourceComponent
from engine.audio.cue import SoundCue, load_cue, parse_cue
from engine.audio.mixer import Mixer
from engine.audio.music import (
    MusicPlayer,
    MusicPlaylist,
    MusicTrack,
    load_playlist,
    parse_playlist,
)
from engine.audio.spatial import Attenuation, attenuation
from engine.audio.system import AudioSystem, load_audio_settings, save_audio_settings

__all__ = [
    "BUSES",
    "CLIP_CACHE_LIMIT",
    "AudioBackendError",
    "AudioClip",
    "AudioDecodeError",
    "AudioListenerComponent",
    "AudioSourceComponent",
    "AudioSystem",
    "Attenuation",
    "ClipCache",
    "IAudioBackend",
    "Mixer",
    "MusicPlayer",
    "MusicPlaylist",
    "MusicTrack",
    "NullBackend",
    "SoftwareBackend",
    "SoundCue",
    "VoiceHandle",
    "attenuation",
    "create_backend",
    "decode_audio",
    "decode_wav",
    "load_audio_clip",
    "load_audio_settings",
    "load_cue",
    "load_playlist",
    "parse_cue",
    "parse_playlist",
    "save_audio_settings",
]
