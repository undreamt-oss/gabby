# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026-present Gabby Contributors.

"""Small cross-platform checks for files opened from host-managed paths."""

from __future__ import annotations

import os
import stat


def opened_file_matches_path(
    path_metadata: os.stat_result,
    opened_metadata: os.stat_result,
) -> bool:
    """Check that an opened regular file matches its pre-open path snapshot.

    POSIX file IDs provide a stable identity check. Windows runners can report differing or
    unavailable IDs between path stat and handle stat, so use size and timestamps there.
    """
    if not stat.S_ISREG(opened_metadata.st_mode):
        return False

    if os.name == "nt":
        return (
            path_metadata.st_size == opened_metadata.st_size
            and path_metadata.st_mtime_ns == opened_metadata.st_mtime_ns
            and path_metadata.st_ctime_ns == opened_metadata.st_ctime_ns
        )

    return (
        path_metadata.st_dev == opened_metadata.st_dev
        and path_metadata.st_ino == opened_metadata.st_ino
    )
