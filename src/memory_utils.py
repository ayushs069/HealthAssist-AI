import ctypes
import os
from typing import Callable, Optional


def get_available_memory_gb() -> Optional[float]:
    """
    Return currently available physical memory in GB.
    Falls back to OS-specific methods when optional deps are unavailable.
    """
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / (1024**3)
    except Exception:
        pass

    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float(status.ullAvailPhys) / (1024**3)
        return None

    meminfo = "/proc/meminfo"
    if os.path.exists(meminfo):
        try:
            with open(meminfo, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        kb = float(line.split()[1])
                        return kb / (1024**2)
        except Exception:
            return None

    return None


def warn_if_low_memory(
    min_available_gb: float = 2.0,
    context: str = "runtime",
    emitter: Callable[[str], None] = print,
) -> Optional[float]:
    """
    Emit warning when available RAM drops below baseline.
    Returns available memory in GB when measurable, else None.
    """
    available_gb = get_available_memory_gb()
    if available_gb is None:
        return None

    if available_gb < float(min_available_gb):
        emitter(
            f"WARNING: Low available RAM in {context}: "
            f"{available_gb:.2f} GB (baseline {float(min_available_gb):.2f} GB)."
        )

    return available_gb
