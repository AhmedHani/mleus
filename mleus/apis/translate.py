# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The mleus Project, https://github.com/AhmedHani/mleus"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Ahmed H. Al-Ghidani"


from googletrans import Translator


__translator = Translator()


def translate(text, src='en', dest='ar'):
    return __translator.translate(text, src=src, dest=dest).text
