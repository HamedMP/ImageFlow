# Copyright 2016 Hamed MP. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""
  Simple library to read all PNG and JPG/JPEG images in a directory
  with TensorFlow buil-in functions in multi-thread way to boost speed.

  Supported formats by TensorFlow are: PNG, JPG/JPEG

  Hamed MP
  Github: @hamedmp
  Twitter: @TheHamedMP
"""

from .imageflow import *
from .reader import read_and_decode

__author__ = 'HANEL'
